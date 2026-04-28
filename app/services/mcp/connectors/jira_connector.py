"""
Jira MCP Connector.

지원 action_type:
  - create_jira_issue
  - create_voc_issue  (Jira 백로그 생성으로 처리)

── real mode env ─────────────────────────────────────────────────────────────
  JIRA_MCP_REAL=true    real mode 활성화
  JIRA_PROJECT_KEY      Jira 프로젝트 키 (필수, 예: VOC)
  JIRA_ISSUE_TYPE       이슈 타입 (필수, 예: Bug)
  JIRA_BASE_URL         Jira 인스턴스 URL (선택)
  JIRA_EMAIL            인증 이메일 (선택)
  JIRA_API_TOKEN        API 토큰 (선택)
  JIRA_MCP_SERVER_URL   MCP 서버 URL (선택)

── mock mode ─────────────────────────────────────────────────────────────────
  status: success
  external_id: jira-mock-{call_id}
  result: {project_key, issue_type, summary, description, labels, mock}

── real mode 설정 부족 ────────────────────────────────────────────────────────
  status: skipped
  error: "jira_mcp_connector_not_configured"
"""
from __future__ import annotations

import os

from app.services.mcp.connectors.base import BaseMCPConnector
from app.utils.logger import get_logger

logger = get_logger(__name__)


class JiraConnector(BaseMCPConnector):
    connector_name = "jira"
    _real_mode_env = "JIRA_MCP_REAL"
    _required_config = ("JIRA_PROJECT_KEY", "JIRA_ISSUE_TYPE")
    _oauth_provider_name = "jira"

    async def execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        logger.info(
            "JiraConnector call_id=%s action_type=%s real_mode=%s tenant_oauth=%s",
            call_id, action_type, self.is_real_mode(), self._use_tenant_oauth(),
        )

        # tenant OAuth 우선 시도
        if self._use_tenant_oauth() and tenant_id:
            result = await self._try_tenant_token(tenant_id)
            if result["error"] != "tenant_integration_not_connected" or not self._allow_env_fallback():
                return result

        if not self.is_real_mode():
            return self._mock(params, call_id)

        ok, err = self.validate_config()
        if not ok:
            logger.warning("JiraConnector: config 부족 call_id=%s err=%s", call_id, err)
            return self._skipped("jira_mcp_connector_not_configured")

        return await self._execute_real(action_type, params, call_id=call_id)

    def _mock(self, params: dict, call_id: str) -> dict:
        project_key = params.get("project_key", "VOC")
        issue_type = params.get("issue_type", "Bug")
        return self._success(
            external_id=f"jira-mock-{call_id}",
            result={
                "project_key": project_key,
                "issue_type": issue_type,
                "summary": params.get("summary", params.get("summary_short", "")),
                "description": params.get("description", params.get("reason", "")),
                "labels": params.get("labels", []),
                "mock": True,
            },
        )

    async def _execute_real(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
    ) -> dict:
        # TODO: 실제 Jira API / MCP 서버 연동 구현
        # project_key = os.getenv("JIRA_PROJECT_KEY")
        # issue_type = os.getenv("JIRA_ISSUE_TYPE")
        # base_url = os.getenv("JIRA_BASE_URL")
        # email = os.getenv("JIRA_EMAIL")
        # token = os.getenv("JIRA_API_TOKEN")
        # server_url = os.getenv("JIRA_MCP_SERVER_URL")
        logger.warning(
            "JiraConnector: real mode TODO — skipped call_id=%s action_type=%s",
            call_id, action_type,
        )
        return self._skipped("jira_mcp_real_not_implemented")
