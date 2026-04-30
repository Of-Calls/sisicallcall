"""
Jira MCP Connector.

지원 action_type:
  - create_jira_issue
  - create_voc_issue  (Jira 백로그 생성으로 처리)

── 실행 정책 ─────────────────────────────────────────────────────────────────
  JIRA_MCP_REAL=false (기본) → 항상 mock 반환  (MCP_USE_TENANT_OAUTH 값 무관)
  JIRA_MCP_REAL=true  + MCP_USE_TENANT_OAUTH=false → skipped("tenant_oauth_required")
  JIRA_MCP_REAL=true  + MCP_USE_TENANT_OAUTH=true + tenant_id 없음 → skipped("tenant_oauth_required")
  JIRA_MCP_REAL=true  + MCP_USE_TENANT_OAUTH=true + integration 없음 → skipped("tenant_integration_not_connected")
  JIRA_MCP_REAL=true  + MCP_USE_TENANT_OAUTH=true + workspace 미선택 → skipped("jira_workspace_not_selected")
  JIRA_MCP_REAL=true  + MCP_USE_TENANT_OAUTH=true + integration 있음 → Jira API 호출

── env API token fallback 없음 ─────────────────────────────────────────────
  Jira real mode는 tenant OAuth 전용이다.
  JIRA_EMAIL / JIRA_API_TOKEN / Basic Auth / MCP_ALLOW_ENV_FALLBACK 사용 금지.

── API endpoint 결정 ──────────────────────────────────────────────────────
  cloud_id 탐색 순서:
    1) integration.metadata["cloud_id"]
    2) integration.metadata["cloudId"]
    3) integration.external_workspace_id   ← Atlassian OAuth 저장 시 cloud_id

  cloud_id → https://api.atlassian.com/ex/jira/{cloud_id}/rest/api/3/issue
  없으면  → skipped("jira_workspace_not_selected")

── 필수 env ──────────────────────────────────────────────────────────────
  JIRA_PROJECT_KEY  — 이슈 생성 프로젝트 키 (기본값 "VOC")
  JIRA_ISSUE_TYPE   — 이슈 타입 (기본값 "Task")
  JIRA_ISSUE_TYPE_ID — 이슈 타입 ID (설정 시 JIRA_ISSUE_TYPE 대신 사용)

── mock mode ─────────────────────────────────────────────────────────────────
  status: success
  external_id: jira-mock-{call_id}
  result: {project_key, issue_type, summary, description, labels, mock}
"""
from __future__ import annotations

import os
from datetime import datetime

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

        # mock 판단이 항상 먼저 — JIRA_MCP_REAL=false이면 무조건 mock
        if not self.is_real_mode():
            return self._mock(params, call_id)

        # real mode: tenant OAuth 필수 (env API token fallback 없음)
        if not self._use_tenant_oauth() or not tenant_id:
            logger.warning(
                "JiraConnector: tenant OAuth required call_id=%s tenant_id=%r",
                call_id, tenant_id,
            )
            return self._skipped("tenant_oauth_required")

        return await self._oauth_execute(action_type, params, call_id=call_id, tenant_id=tenant_id)

    # ── tenant OAuth 실행 ─────────────────────────────────────────────────────

    async def _oauth_execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str,
    ) -> dict:
        from app.models.tenant_integration import IntegrationStatus
        from app.repositories.tenant_integration_repo import get_integration
        from app.services.oauth.token_crypto import decrypt_token

        integration = get_integration(tenant_id, self._oauth_provider_name)

        if integration is None or integration.status == IntegrationStatus.disconnected:
            return self._skipped("tenant_integration_not_connected")

        # 만료 체크
        if integration.expires_at and integration.expires_at < datetime.utcnow():
            if integration.refresh_token_encrypted:
                refreshed = await self._refresh_tenant_token(integration)
                if refreshed:
                    integration = refreshed
                else:
                    return self._skipped("tenant_token_expired_refresh_failed")
            else:
                return self._skipped("tenant_token_expired_no_refresh")

        try:
            access_token = decrypt_token(integration.access_token_encrypted or "")
        except Exception:
            logger.error(
                "JiraConnector: token 복호화 실패 call_id=%s tenant_id=%s",
                call_id, tenant_id,
            )
            return self._failed("tenant_token_decryption_failed")

        # API endpoint 결정
        meta: dict = getattr(integration, "metadata", None) or {}
        if meta.get("workspace_selection_required") is True:
            logger.warning(
                "JiraConnector: Jira workspace not selected call_id=%s tenant_id=%s",
                call_id,
                tenant_id,
            )
            return self._skipped("jira_workspace_not_selected")

        cloud_id = (
            meta.get("cloud_id")
            or meta.get("cloudId")
            or getattr(integration, "external_workspace_id", None)
            or ""
        )
        if not cloud_id:
            logger.warning(
                "JiraConnector: Jira workspace not selected call_id=%s tenant_id=%s",
                call_id, tenant_id,
            )
            return self._skipped("jira_workspace_not_selected")

        api_url = f"https://api.atlassian.com/ex/jira/{cloud_id}/rest/api/3/issue"

        project_key = os.getenv("JIRA_PROJECT_KEY", "VOC")
        issue_type = os.getenv("JIRA_ISSUE_TYPE", "Task")
        issue_type_id = os.getenv("JIRA_ISSUE_TYPE_ID", "").strip()
        summary = (
            params.get("summary")
            or params.get("title")
            or params.get("summary_short")
            or "[시시콜콜] VOC 후속 이슈"
        )
        description_text = (
            params.get("description")
            or params.get("reason")
            or params.get("summary_short")
            or ""
        )
        labels = params.get("labels") or ["sisicallcall", "post-call"]

        return await self._create_issue(
            access_token,
            api_url=api_url,
            project_key=project_key,
            issue_type=issue_type,
            issue_type_id=issue_type_id,
            summary=summary,
            description_text=description_text,
            labels=labels,
            call_id=call_id,
        )

    # ── Jira issue create ────────────────────────────────────────────────────

    async def _create_issue(
        self,
        token: str,
        *,
        api_url: str,
        project_key: str,
        issue_type: str,
        issue_type_id: str,
        summary: str,
        description_text: str,
        labels: list,
        call_id: str,
    ) -> dict:
        """Jira Cloud REST API v3 issue create를 호출한다. token은 로그에 출력하지 않는다."""
        import httpx

        issuetype_field = {"id": issue_type_id} if issue_type_id else {"name": issue_type}
        body = {
            "fields": {
                "project": {"key": project_key},
                "issuetype": issuetype_field,
                "summary": summary,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": description_text}],
                        }
                    ],
                },
                "labels": labels,
            }
        }

        try:
            logger.debug(
                "JiraConnector: create issue request call_id=%s project=%s issue_type=%s issue_type_id=%s summary=%s labels=%s",
                call_id,
                project_key,
                issue_type,
                issue_type_id or "",
                summary,
                labels,
            )
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    api_url,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    timeout=15.0,
                )

            if resp.status_code not in (200, 201):
                body_preview = (getattr(resp, "text", "") or "")[:1000]
                logger.error(
                    "JiraConnector: HTTP 오류 call_id=%s status=%d body=%s",
                    call_id,
                    resp.status_code,
                    body_preview,
                )
                return self._failed(f"jira_http_error:{resp.status_code}")

            data = resp.json()
            issue_key = data.get("key", "")
            issue_id = data.get("id", "")
            issue_self = data.get("self", "")
            logger.info(
                "JiraConnector: 이슈 생성 완료 call_id=%s issue_key=%s",
                call_id, issue_key,
            )
            return self._success(
                external_id=issue_key,
                result={"issue_key": issue_key, "issue_id": issue_id, "self": issue_self},
            )

        except Exception as exc:
            logger.error(
                "JiraConnector: 예외 call_id=%s err=%s",
                call_id, type(exc).__name__,
            )
            return self._failed(f"jira_exception:{type(exc).__name__}")

    # ── mock ─────────────────────────────────────────────────────────────────

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
