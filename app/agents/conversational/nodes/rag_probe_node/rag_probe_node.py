"""RAG probe — cache miss 직후, intent_router 진입 전.

ChromaDB top_k=3 검색을 1회 가볍게 실행해 router LLM 에게 RAG 강/약 신호를
state["rag_probe"] 로 노출한다. router 가 이 신호를 보고 더 정확하게 intent
분류 가능 (특히 모호 발화 + 강 RAG 신호, 명확 발화 + RAG miss 케이스).

신호 (state["rag_probe"]):
    {
        "top_distance": float,           # cosine distance, 0~2 (낮을수록 유사)
        "matched_keywords": list[str],   # query 와 chunk.llm_keywords substring 매칭
        "top_topic": str,                # chunk.llm_topic (clarify 유도질문용)
        "top_title": str,                # chunk.llm_title (디버그)
        "top_chunk_id": str,             # 디버그/추적용
    }
오류·미보유 시 state["rag_probe"] = None.

검색 자체는 faq_branch_node 가 다시 수행 (top_k=8, hybrid score, distance filter).
즉 RAG 검색이 cache miss turn 마다 2회 발생 — 신호용/답변용. 후속 최적화 후보:
  - faq_branch 가 probe 결과 (top_k=3) 재사용해 추가 query 생략

부수 책임 (architect Phase B+C 병합) — stall 발화 트리거.
graph 가 cache miss 분기로 들어왔을 때만 본 노드가 실행되므로, probe 결과를
보고 차등 멘트(audio_field) 를 선택해 stall 발화를 fire-and-forget 으로 spawn.
push_stall 의 '턴당 1회' 가드가 call.py 의 1.5초 fallback scheduler 와 중복
방출 차단.
"""
import asyncio
from typing import Optional

from app.agents.conversational.state import CallState
from app.services.rag.base import BaseRAGService
from app.services.rag.chroma import ChromaRAGService
from app.services.tts.channel import tts_channel
from app.utils.logger import get_logger

logger = get_logger(__name__)

_rag: BaseRAGService = ChromaRAGService()

PROBE_TOP_K = 3
PROBE_TIMEOUT_SEC = 1.0  # 5초 hardcut 안전 마진. ChromaDB local query ~30~80ms 추정.

# 차등 stall (architect Phase C) — 명백한 신호일 때만 audio_field 변경, 그 외 general.
_STALL_DIFF_DISTANCE_THRESHOLD = 0.4
_STALL_DIFF_MIN_KEYWORDS = 2
# top_topic 키워드 → audio_field 매핑. 명백한 task/auth 키워드만 등록.
# 매칭 없으면서 신호는 강한 경우 → "faq" (정보 안내) 로 fallback.
_TOPIC_TO_FIELD: list[tuple[str, str]] = [
    ("예약", "task"), ("접수", "task"), ("신청", "task"),
    ("변경", "task"), ("취소", "task"), ("조회", "task"), ("주문", "task"),
    ("인증", "auth"), ("본인", "auth"), ("회원", "auth"),
]
_STALL_FALLBACK_TEXT = "잠시만요, 확인해 드리겠습니다."


def _select_stall_field(probe: Optional[dict], stall_messages: dict) -> str:
    """rag_probe 신호 + tenant 가용 stall_messages 키 기반 audio_field 선택.

    보수 규칙: top_distance < 0.4 AND matched_keywords ≥ 2 일 때만 차등.
    매핑 결과 키가 stall_messages 에 없으면 general 로 fallback.
    """
    if not probe:
        return "general"
    distance = probe.get("top_distance")
    matched = probe.get("matched_keywords") or []
    topic = probe.get("top_topic") or ""

    if (
        distance is None
        or distance >= _STALL_DIFF_DISTANCE_THRESHOLD
        or len(matched) < _STALL_DIFF_MIN_KEYWORDS
    ):
        return "general"

    for kw, field in _TOPIC_TO_FIELD:
        if kw in topic:
            return field if field in stall_messages else "general"
    # 매핑 없지만 신호는 강함 → 정보 안내 류 (faq)
    if topic and "faq" in stall_messages:
        return "faq"
    return "general"


def _spawn_stall(state: CallState, audio_field: str) -> None:
    """stall 발화 spawn — fire-and-forget. push_stall '턴당 1회' 가드가 중복 차단."""
    stall_messages = state.get("stall_messages") or {}
    text = (
        stall_messages.get(audio_field)
        or stall_messages.get("general")
        or _STALL_FALLBACK_TEXT
    )
    asyncio.create_task(
        tts_channel.push_stall(
            call_id=state["call_id"],
            text=text,
            audio_field=audio_field,
        ),
        name=f"stall:{state['call_id']}:{audio_field}",
    )


async def rag_probe_node(state: CallState) -> dict:
    call_id = state["call_id"]
    query_embedding = state.get("query_embedding") or []
    stall_messages = state.get("stall_messages") or {}

    if not query_embedding:
        # 임베딩 부재 → probe skip, stall 은 general 로 즉시 발화
        _spawn_stall(state, "general")
        return {"rag_probe": None}

    try:
        results = await asyncio.wait_for(
            _rag.search_with_meta(
                query_embedding=query_embedding,
                tenant_id=state["tenant_id"],
                top_k=PROBE_TOP_K,
            ),
            timeout=PROBE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning("rag_probe timeout call_id=%s", call_id)
        _spawn_stall(state, "general")
        return {"rag_probe": None}
    except Exception as e:
        logger.error("rag_probe error call_id=%s: %s", call_id, e)
        _spawn_stall(state, "general")
        return {"rag_probe": None}

    if not results:
        _spawn_stall(state, "general")
        return {"rag_probe": None}

    top = results[0]
    meta = top.get("metadata") or {}
    query_text = state.get("normalized_text") or ""
    keywords = [k.strip() for k in (meta.get("llm_keywords") or "").split(",") if k.strip()]
    matched = [kw for kw in keywords if kw in query_text]

    probe = {
        "top_distance": top.get("distance"),
        "matched_keywords": matched,
        "top_topic": meta.get("llm_topic", ""),
        "top_title": meta.get("llm_title", ""),
        "top_chunk_id": top.get("id", ""),
    }

    field = _select_stall_field(probe, stall_messages)
    _spawn_stall(state, field)

    logger.info(
        "rag_probe call_id=%s distance=%.3f matched=%s topic=%r stall_field=%s",
        call_id,
        probe["top_distance"] if probe["top_distance"] is not None else -1.0,
        matched, probe["top_topic"], field,
    )
    return {"rag_probe": probe}
