from __future__ import annotations

from app.utils.logger import get_logger

logger = get_logger(__name__)


class CalendarAction:
    """Calendar MCP Connector를 통해 콜백 일정을 예약한다."""

    async def execute(self, action: dict, *, call_id: str, tenant_id: str = "") -> dict:
        from app.services.mcp.client import mcp_client

        logger.info("CalendarAction call_id=%s action_type=%s", call_id, action.get("action_type"))
        return await mcp_client.call_tool(
            "calendar",
            action.get("action_type", "schedule_callback"),
            action.get("params", {}),
            call_id=call_id,
            tenant_id=tenant_id,
        )
