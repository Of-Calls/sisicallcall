"""
Slack MCP Connector.

지원 action_type:
  - send_slack_alert

── tenant OAuth mode (MCP_USE_TENANT_OAUTH=true) ───────────────────────────
  slack integration token을 사용해 Slack Web API chat.postMessage를 호출.
  tenant_id 기준으로 TenantIntegration을 조회하고 Fernet 복호화한 access_token으로
  Authorization: Bearer 헤더를 구성한다.
  token이 없으면 skipped, API ok=false 시 failed, HTTP 오류 시 failed, 성공 시 success.
  access_token 원문은 로그에 출력하지 않는다.

── real mode env (.env SLACK_BOT_TOKEN) ────────────────────────────────────
  SLACK_MCP_REAL=true + SLACK_BOT_TOKEN 설정 시 사용.

── mock mode ─────────────────────────────────────────────────────────────────
  status: success
  external_id: slack-mock-{call_id}
  result: {channel, message, mock}

── channel 우선순위 ─────────────────────────────────────────────────────────
  params.channel > params.channel_id
  > (params.channel_type == "critical" → SLACK_CRITICAL_CHANNEL)
  > SLACK_ALERT_CHANNEL

── message/text 우선순위 ────────────────────────────────────────────────────
  params.message > params.text > params.summary_short > "Post-call alert"
"""
from __future__ import annotations

import os
from datetime import datetime

from app.services.mcp.connectors.base import BaseMCPConnector
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SlackConnector(BaseMCPConnector):
    connector_name = "slack"
    _real_mode_env = "SLACK_MCP_REAL"
    _required_config = ("SLACK_ALERT_CHANNEL",)
    _oauth_provider_name = "slack"

    async def execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        logger.info(
            "SlackConnector call_id=%s action_type=%s real_mode=%s tenant_oauth=%s",
            call_id, action_type, self.is_real_mode(), self._use_tenant_oauth(),
        )

        # tenant OAuth 우선 처리 (CalendarConnector 패턴)
        if self._use_tenant_oauth() and tenant_id:
            return await self._oauth_execute(action_type, params, call_id=call_id, tenant_id=tenant_id)

        if not self.is_real_mode():
            return self._mock(params, call_id)

        ok, err = self.validate_config()
        if not ok:
            logger.warning("SlackConnector: config 부족 call_id=%s err=%s", call_id, err)
            return self._skipped("slack_mcp_connector_not_configured")

        return await self._execute_env_real(action_type, params, call_id=call_id)

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
            if self._allow_env_fallback():
                return self._mock(params, call_id) if not self.is_real_mode() else await self._execute_env_real(action_type, params, call_id=call_id)
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
                "SlackConnector: token 복호화 실패 call_id=%s tenant_id=%s",
                call_id, tenant_id,
            )
            return self._failed("tenant_token_decryption_failed")

        return await self._post_message(access_token, params, call_id=call_id)

    # ── .env BOT_TOKEN 실행 ───────────────────────────────────────────────────

    async def _execute_env_real(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
    ) -> dict:
        bot_token = os.getenv("SLACK_BOT_TOKEN")
        if not bot_token:
            logger.warning("SlackConnector: SLACK_BOT_TOKEN 없음 call_id=%s", call_id)
            return self._skipped("slack_bot_token_not_configured")
        return await self._post_message(bot_token, params, call_id=call_id)

    # ── Slack chat.postMessage ────────────────────────────────────────────────

    async def _post_message(self, token: str, params: dict, *, call_id: str) -> dict:
        """Slack Web API chat.postMessage를 호출한다. token은 로그에 출력하지 않는다."""
        import httpx

        channel = (
            params.get("channel")
            or params.get("channel_id")
            or (
                os.getenv("SLACK_CRITICAL_CHANNEL")
                if params.get("channel_type") == "critical"
                else None
            )
            or os.getenv("SLACK_ALERT_CHANNEL", "#alerts")
        )
        text = (
            params.get("message")
            or params.get("text")
            or params.get("summary_short")
            or "Post-call alert"
        )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://slack.com/api/chat.postMessage",
                    json={"channel": channel, "text": text},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=15.0,
                )

            if resp.status_code != 200:
                logger.error(
                    "SlackConnector: HTTP 오류 call_id=%s status=%d",
                    call_id, resp.status_code,
                )
                return self._failed(f"slack_http_error:{resp.status_code}")

            data = resp.json()
            if not data.get("ok"):
                error_code = data.get("error", "unknown_slack_error")
                logger.error(
                    "SlackConnector: API ok=false call_id=%s error=%s",
                    call_id, error_code,
                )
                return self._failed(f"slack_api_error:{error_code}")

            ts = data.get("ts", "")
            ch = data.get("channel", channel)
            logger.info(
                "SlackConnector: 메시지 전송 완료 call_id=%s channel=%s ts=%s",
                call_id, ch, ts,
            )
            return self._success(
                external_id=f"{ch}:{ts}",
                result={
                    "channel": ch,
                    "ts": ts,
                    "message": data.get("message", {}),
                },
            )

        except Exception as exc:
            logger.error(
                "SlackConnector: 예외 call_id=%s err=%s",
                call_id, type(exc).__name__,
            )
            return self._failed(f"slack_exception:{type(exc).__name__}")

    # ── mock ─────────────────────────────────────────────────────────────────

    def _mock(self, params: dict, call_id: str) -> dict:
        channel = params.get("channel", os.getenv("SLACK_ALERT_CHANNEL", "#alerts"))
        message = params.get("message", f"[ALERT] call_id={call_id}")
        return self._success(
            external_id=f"slack-mock-{call_id}",
            result={
                "channel": channel,
                "message": message,
                "mock": True,
            },
        )
