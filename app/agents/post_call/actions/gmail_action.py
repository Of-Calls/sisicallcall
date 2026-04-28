from __future__ import annotations

from app.utils.logger import get_logger

logger = get_logger(__name__)


class GmailAction:
    """Gmail MCP Connector를 통해 이메일을 전송한다."""

    async def execute(self, action: dict, *, call_id: str, tenant_id: str = "") -> dict:
        from app.services.mcp.client import mcp_client

        logger.info("GmailAction call_id=%s action_type=%s", call_id, action.get("action_type"))
        return await mcp_client.call_tool(
            "gmail",
            action.get("action_type", "send_manager_email"),
            action.get("params", {}),
            call_id=call_id,
            tenant_id=tenant_id,
        )
