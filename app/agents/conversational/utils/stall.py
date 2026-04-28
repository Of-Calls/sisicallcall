"""_run_with_stall — LLM 응답 hardcut helper.

stall 발화 책임은 `app/api/v1/call.py:_schedule_stall_task` (graph 진입 직후 1.5초
background scheduler) 로 이관됨 (2026-04-28). 본 헬퍼는 hardcut 만 담당.

hardcut 시 rag_results[0] 을 그대로 fallback 으로 쓰면 raw markdown 청크가
TTS 로 흘러가 "##" 같은 마커가 음성으로 발음됨 → `_sanitize_chunk_for_voice`
로 정제 후 반환.
"""
import asyncio
import re
from typing import Awaitable, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

FALLBACK_MESSAGE = "확인이 어려워 담당자에게 연결해 드리겠습니다."

# markdown 마커 + 표 구분자 + table pipe — TTS 가 "샾", "별표" 등으로 읽지 않게 제거.
_MD_NOISE_RE = re.compile(r'[#■▶☑※*`|]+')
_FALLBACK_MAX_CHARS = 200


def _sanitize_chunk_for_voice(text: str, max_chars: int = _FALLBACK_MAX_CHARS) -> str:
    """RAG raw chunk 를 음성 친화 텍스트로 정제. markdown 마커 제거 + 길이 제한."""
    cleaned = _MD_NOISE_RE.sub('', text or '')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1].rstrip() + '…'
    return cleaned


async def _run_with_stall(
    *,
    coro: Awaitable[str],
    call_id: str,
    hardcut_sec: float,
    rag_results: Optional[list[str]] = None,
    fallback_text: str = FALLBACK_MESSAGE,
) -> tuple[str, bool]:
    """(response_text, is_timeout) 튜플 반환. hardcut_sec 초과 시 fallback."""
    llm_task = asyncio.create_task(coro)
    try:
        response_text = await asyncio.wait_for(
            asyncio.shield(llm_task), timeout=hardcut_sec
        )
        return (response_text or "").strip(), False
    except asyncio.TimeoutError:
        logger.warning("stall helper hardcut call_id=%s", call_id)
        llm_task.cancel()
        if rag_results:
            return _sanitize_chunk_for_voice(rag_results[0]), True
        return fallback_text, True
    except Exception as e:
        logger.error("stall helper error call_id=%s: %s", call_id, e)
        llm_task.cancel()
        return fallback_text, False
