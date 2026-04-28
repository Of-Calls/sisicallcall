from app.agents.conversational.state import CallState
from app.services.cache.semantic_cache import SemanticCacheService
from app.utils.logger import get_logger

# 응답 생성 후 Semantic Cache 저장 책임 노드 (TTS 직전 단계).
# state 변경 없이 read-only 로 동작. 실패해도 본 턴 응답 흐름에 영향 없도록 예외 흡수.
#
# 저장 금지 6조건 (CLAUDE.md + 의미 fallback 확장):
# - is_timeout=True (타임아웃 폴백 응답)
# - is_fallback=True (RAG miss / LLM 고정 fallback — 브랜치 노드가 명시)
# - response_path == "escalation" (에스컬레이션 응답)
# - response_path == "cache" (이미 캐시에서 온 응답 재저장 방지)
# - response_path == "clarify" (역질문은 답변이 아니므로 캐시 대상 아님)
# - response_path == "repeat" (이전 AI 응답 재생 — 캐시 self-reference 방지)
# - reviewer_verdict == "revise" (Reviewer 가 수정한 응답)

logger = get_logger(__name__)
_cache_service = SemanticCacheService()

_BLOCKED_RESPONSE_PATHS = {"escalation", "cache", "clarify", "repeat"}


def _should_store(state: CallState) -> bool:
    if state.get("is_timeout"):
        return False
    if state.get("is_fallback"):
        return False
    if state.get("response_path") in _BLOCKED_RESPONSE_PATHS:
        return False
    if state.get("reviewer_verdict") == "revise":
        return False
    if not state.get("response_text"):
        return False
    if not state.get("query_embedding"):
        return False
    return True


async def cache_store_node(state: CallState) -> dict:
    if not _should_store(state):
        return {"cache_stored": False}

    try:
        await _cache_service.store(
            text=state["normalized_text"],
            tenant_id=state["tenant_id"],
            embedding=state["query_embedding"],
            response_text=state["response_text"],
            cache_source=state.get("response_path", "unknown"),
        )
        return {"cache_stored": True}
    except Exception as e:
        logger.error("cache store failed call_id=%s: %s", state["call_id"], e)
        return {"cache_stored": False}
