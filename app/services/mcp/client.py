"""
MCPClient — MCP Connector Registry & Router.

PostCallAgent → ActionExecutor → MCPClient → Connector → 외부 도구

역할:
  - tool_name → Connector 매핑 관리
  - call_tool()로 단일 진입점 제공
  - Connector 예외를 failed 결과로 변환 (호출자에게 예외 전파하지 않음)
  - 알 수 없는 tool → failed 반환

기본 등록 tool:
  gmail, calendar, company_db, jira, slack
  (internal_dashboard는 ActionExecutor의 InternalDashboardAction이 직접 처리)

반환 형식:
  {
    "status":      "success" | "failed" | "skipped",
    "external_id": "..." | None,
    "result":      {...},
    "error":       None | "..."
  }
"""
from __future__ import annotations

from app.services.mcp.connectors.base import BaseMCPConnector
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MCPClient:
    """MCP Connector를 등록하고 tool 호출을 라우팅한다."""

    def __init__(self) -> None:
        self._registry: dict[str, BaseMCPConnector] = {}

    # ── Registry 관리 ─────────────────────────────────────────────────────────

    def register_connector(self, tool_name: str, connector: BaseMCPConnector) -> None:
        """tool_name에 connector를 등록한다. 이미 존재하면 덮어쓴다."""
        self._registry[tool_name] = connector
        logger.debug(
            "MCPClient: registered connector tool=%s connector=%s",
            tool_name, type(connector).__name__,
        )

    def get_connector(self, tool_name: str) -> BaseMCPConnector | None:
        """등록된 connector를 반환한다. 없으면 None."""
        return self._registry.get(tool_name)

    def registered_tools(self) -> list[str]:
        """현재 등록된 tool 이름 목록."""
        return list(self._registry.keys())

    # ── 실행 진입점 ───────────────────────────────────────────────────────────

    async def call_tool(
        self,
        tool_name: str,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        """tool_name에 해당하는 connector를 찾아 action을 실행한다.

        - 알 수 없는 tool → failed
        - connector 예외 → failed
        - 정상 실행 → connector 반환값 그대로 반환
        """
        connector = self._registry.get(tool_name)
        if connector is None:
            logger.warning(
                "MCPClient: unknown tool=%r call_id=%s action_type=%s",
                tool_name, call_id, action_type,
            )
            return {
                "status": "failed",
                "external_id": None,
                "result": {},
                "error": f"unknown tool: {tool_name!r}",
            }

        try:
            result = await connector.execute(
                action_type,
                params,
                call_id=call_id,
                tenant_id=tenant_id,
            )
            logger.debug(
                "MCPClient: call_tool tool=%s action_type=%s status=%s call_id=%s",
                tool_name, action_type, result.get("status"), call_id,
            )
            return result
        except Exception as exc:
            logger.error(
                "MCPClient: connector exception tool=%s call_id=%s err=%s",
                tool_name, call_id, exc,
            )
            return {
                "status": "failed",
                "external_id": None,
                "result": {},
                "error": str(exc),
            }


# ── 모듈 레벨 singleton ───────────────────────────────────────────────────────


def _build_default_client() -> MCPClient:
    """기본 5개 connector를 등록한 MCPClient 인스턴스를 반환한다."""
    from app.services.mcp.connectors.gmail_connector import GmailConnector
    from app.services.mcp.connectors.calendar_connector import CalendarConnector
    from app.services.mcp.connectors.company_db_connector import CompanyDBConnector
    from app.services.mcp.connectors.jira_connector import JiraConnector
    from app.services.mcp.connectors.slack_connector import SlackConnector

    client = MCPClient()
    client.register_connector("gmail", GmailConnector())
    client.register_connector("calendar", CalendarConnector())
    client.register_connector("company_db", CompanyDBConnector())
    client.register_connector("jira", JiraConnector())
    client.register_connector("slack", SlackConnector())
    return client


mcp_client: MCPClient = _build_default_client()
