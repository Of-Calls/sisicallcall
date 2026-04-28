from __future__ import annotations

from app.utils.logger import get_logger

logger = get_logger(__name__)


class SlackAction:
    """Slack MCP Connector를 통해 알림 메시지를 전송한다."""

    async def execute(self, action: dict, *, call_id: str, tenant_id: str = "") -> dict:
        from app.services.mcp.client import mcp_client

        logger.info("SlackAction call_id=%s action_type=%s", call_id, action.get("action_type"))
        return await mcp_client.call_tool(
            "slack",
            action.get("action_type", "send_slack_alert"),
            action.get("params", {}),
            call_id=call_id,
            tenant_id=tenant_id,
        )
