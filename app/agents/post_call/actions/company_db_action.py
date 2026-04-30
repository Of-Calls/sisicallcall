from __future__ import annotations

from app.utils.logger import get_logger

logger = get_logger(__name__)


class CompanyDBAction:
    """Company DB MCP Connector를 통해 VOC 이슈를 등록한다."""

    async def execute(self, action: dict, *, call_id: str, tenant_id: str = "") -> dict:
        from app.services.mcp.client import mcp_client

        logger.info("CompanyDBAction call_id=%s action_type=%s", call_id, action.get("action_type"))
        return await mcp_client.call_tool(
            "company_db",
            action.get("action_type", "create_voc_issue"),
            action.get("params", {}),
            call_id=call_id,
            tenant_id=tenant_id,
        )
