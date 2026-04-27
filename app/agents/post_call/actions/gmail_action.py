from __future__ import annotations

import os

from app.utils.logger import get_logger

logger = get_logger(__name__)


class GmailAction:
    async def execute(self, action: dict, *, call_id: str, tenant_id: str = "") -> dict:
        params = action.get("params", {})
        logger.info("GmailAction call_id=%s action_type=%s", call_id, action.get("action_type"))

        if os.getenv("MCP_GMAIL_REAL", "").lower() in ("1", "true"):
            return await self._execute_real(params, call_id=call_id)

        return {
            "external_id": f"gmail-mock-{call_id}",
            "status": "success",
            "result": {
                "sent": True,
                "to": params.get("to", "manager@example.com"),
                "subject": params.get("subject", ""),
                "body": params.get("body", ""),
                "mock": True,
            },
        }

    async def _execute_real(self, params: dict, *, call_id: str) -> dict:
        raise NotImplementedError("Gmail 실제 연동은 MCP 구성 후 구현")
