from __future__ import annotations
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GmailAction:
    async def execute(self, action: dict, *, call_id: str) -> dict:
        params = action.get("params", {})
        logger.info("GmailAction(dummy) call_id=%s action_type=%s", call_id, action.get("action_type"))
        # TODO: 실제 Gmail MCP 연동으로 교체
        return {
            "sent": True,
            "to": params.get("to", "manager@example.com"),
            "subject": params.get("subject", ""),
            "mock": True,
        }
