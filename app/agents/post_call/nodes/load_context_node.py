from __future__ import annotations
from app.agents.post_call.state import PostCallAgentState
from app.repositories.call_summary_repo import CallSummaryRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)
_repo = CallSummaryRepository()


async def load_context_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    try:
        # 운영 경로: runner.py가 get_call_context_for_post_call → seed_call_context 순으로
        # 실제 통화 데이터를 미리 주입하므로, 여기서는 seed된 데이터를 그대로 읽는다.
        # context가 없는 경우(DB/Redis 모두 실패) get_call_context는 sample fallback을
        # 반환한다 — 개발/테스트 환경 전용이며 운영 경로에서는 발생하지 않아야 한다.
        ctx = await _repo.get_call_context(call_id)
        logger.info("load_context call_id=%s transcripts=%d", call_id, len(ctx.get("transcripts", [])))
        return {
            "call_metadata": ctx.get("metadata", {}),
            "transcripts": ctx.get("transcripts", []),
            "branch_stats": ctx.get("branch_stats", {}),
        }
    except Exception as exc:
        logger.error("load_context 실패 call_id=%s err=%s", call_id, exc)
        errors = list(state.get("errors", []))  # type: ignore[call-overload]
        errors.append({"node": "load_context", "error": str(exc)})
        return {"call_metadata": {}, "transcripts": [], "branch_stats": {}, "errors": errors}
