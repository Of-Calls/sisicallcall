from __future__ import annotations
from app.agents.post_call.state import PostCallAgentState
from app.repositories.call_summary_repo import CallSummaryRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)
_repo = CallSummaryRepository()


async def load_context_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    try:
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
