from __future__ import annotations

from app.utils.logger import get_logger

logger = get_logger(__name__)


class JiraAction:
    """Jira MCP Connector를 통해 이슈를 생성한다."""

    async def execute(self, action: dict, *, call_id: str, tenant_id: str = "") -> dict:
        from app.services.mcp.client import mcp_client

        logger.info("JiraAction call_id=%s action_type=%s", call_id, action.get("action_type"))
        return await mcp_client.call_tool(
            "jira",
            action.get("action_type", "create_jira_issue"),
            action.get("params", {}),
            call_id=call_id,
            tenant_id=tenant_id,
        )
