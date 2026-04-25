from __future__ import annotations
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CompanyDBAction:
    async def execute(self, action: dict, *, call_id: str) -> dict:
        params = action.get("params", {})
        logger.info("CompanyDBAction(dummy) call_id=%s action_type=%s", call_id, action.get("action_type"))
        # TODO: 실제 CompanyDB MCP 연동으로 교체
        return {
            "created": True,
            "issue_id": f"VOC-MOCK-{call_id}",
            "tier": params.get("tier", "medium"),
            "mock": True,
        }
