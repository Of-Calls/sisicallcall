"""_run_with_stall — LLM 응답 생성 + 대기 멘트 trigger race (RFC 001 v0.2 §6.5).

느린 노드(FAQ/Task/Auth branch, Reviewer, Summary Sync)가 공통으로 사용하는 헬퍼.
노드 본문 try/except 를 이 헬퍼에 위임하고, 노드는 튜플 반환값만 dict 로 감싸 return.

3-phase race 로직:
  Phase 1: delay 초 안에 응답 오면 stall 없이 (response, False) 반환
  Phase 2: delay 경과 → TTSOutputChannel 에 stall push → hardcut-delay 만큼 추가 대기
  Phase 3: hardcut 도달 → (rag_results[0] or fallback_text, True) 반환

예외 처리:
  Phase 1/2 에서 coro 가 raise 하면 catch + log + (fallback_text, False) 반환.
  stall 이 이미 방출됐으면 TTS 큐에 "잠시만요..." + fallback 이 순차 재생된다.
"""
import asyncio
from typing import Awaitable, Optional

from app.services.tts.channel import tts_channel
from app.utils.logger import get_logger

logger = get_logger(__name__)

FALLBACK_MESSAGE = "확인이 어려워 담당자에게 연결해 드리겠습니다."


async def _run_with_stall(
    *,
    coro: Awaitable[str],
    call_id: str,
    stall_msg: str,
    stall_audio_field: str,
    delay: float,
    hardcut_sec: float,
    rag_results: Optional[list[str]] = None,
    fallback_text: str = FALLBACK_MESSAGE,
) -> tuple[str, bool]:
    """(response_text, is_timeout) 튜플 반환.

    Args:
        coro: LLM 응답 코루틴. `_llm.generate(...)` 와 같이 str 반환 예정.
        call_id: TTSOutputChannel 타겟 call ID.
        stall_msg: 대기 멘트 텍스트 (tenant settings 에서 pre-load 된 값).
        stall_audio_field: 사전 캐시 조회 키 ("faq"/"task"/"auth"/"general" 등).
        delay: stall trigger 타이머 (보통 1.0초).
        hardcut_sec: 전체 하드컷 (FAQ 3초 / Task 4초 / Auth 3초 등).
        rag_results: hardcut 시 폴백용 RAG 결과 (FAQ 한정).
        fallback_text: rag_results 없을 때 폴백 문구.

    Returns:
        (response_text, is_timeout): 응답 문자열과 하드컷 도달 여부 플래그.
    """
    # 방어: delay > hardcut 인 비정상 설정에서도 hardcut 가 강제되도록 clamp.
    phase1_timeout = min(delay, hardcut_sec)

    llm_task = asyncio.create_task(coro)

    # ── Phase 1: phase1_timeout 초 안에 응답 오면 stall 없이 즉시 리턴 ────
    try:
        response_text = await asyncio.wait_for(
            asyncio.shield(llm_task), timeout=phase1_timeout
        )
        # LLM 이 None 을 반환할 가능성 방어
        return (response_text or "").strip(), False
    except asyncio.TimeoutError:
        pass  # stall 발동 단계로 진행
    except Exception as e:
        logger.error("stall helper phase1 error call_id=%s: %s", call_id, e)
        llm_task.cancel()
        return fallback_text, False

    # ── Phase 2: stall 방출 후 남은 시간 추가 대기 ───────────────────────
    # delay >= hardcut 이면 여기서 stall 없이 바로 hardcut 으로 가는 게 맞다.
    remaining = max(hardcut_sec - phase1_timeout, 0.0)
    if remaining > 0:
        await tts_channel.push_stall(
            call_id=call_id,
            text=stall_msg,
            audio_field=stall_audio_field,
        )
    try:
        response_text = await asyncio.wait_for(llm_task, timeout=remaining)
        return (response_text or "").strip(), False
    except asyncio.TimeoutError:
        logger.warning("stall helper hardcut call_id=%s", call_id)
        fallback = rag_results[0] if rag_results else fallback_text
        return fallback, True
    except Exception as e:
        logger.error("stall helper phase2 error call_id=%s: %s", call_id, e)
        return fallback_text, False
