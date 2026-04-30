from __future__ import annotations

from app.agents.post_call.completed_call_runner import run_post_call_for_completed_call
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def run_completed_post_call_background(call_id: str, tenant_id: str) -> None:
    """Run completed-call post-call processing without affecting call teardown."""
    try:
        result = await run_post_call_for_completed_call(
            call_id=call_id,
            tenant_id=tenant_id,
            trigger="call_ended",
        )
        logger.info(
            "post-call background complete call_id=%s ok=%s error=%s",
            call_id,
            result.get("ok"),
            result.get("error"),
        )
    except Exception as exc:
        logger.exception(
            "post-call background failed call_id=%s err=%s",
            call_id,
            exc,
        )
