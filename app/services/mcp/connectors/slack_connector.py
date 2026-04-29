"""
Slack MCP Connector.

지원 action_type:
  - send_slack_alert

── 실행 정책 ─────────────────────────────────────────────────────────────────
  SLACK_MCP_REAL=false (기본) → 항상 mock 반환  (MCP_USE_TENANT_OAUTH 값 무관)
  SLACK_MCP_REAL=true  + MCP_USE_TENANT_OAUTH=false → skipped("tenant_oauth_required")
  SLACK_MCP_REAL=true  + MCP_USE_TENANT_OAUTH=true + tenant_id 없음 → skipped("tenant_oauth_required")
  SLACK_MCP_REAL=true  + MCP_USE_TENANT_OAUTH=true + integration 없음 → skipped("tenant_integration_not_connected")
  SLACK_MCP_REAL=true  + MCP_USE_TENANT_OAUTH=true + integration 있음 → chat.postMessage 호출

── SLACK_BOT_TOKEN env fallback 없음 ─────────────────────────────────────────
  Slack real mode는 tenant OAuth 전용이다.
  MCP_ALLOW_ENV_FALLBACK, SLACK_BOT_TOKEN은 Slack 실제 전송에 사용하지 않는다.

── token 우선순위 (real mode) ────────────────────────────────────────────────
  1. metadata["bot"]["bot_access_token"]  — Slack OAuth v1 bot token
  2. metadata["access_token"] (xoxb- 시작) — v2 raw 저장 형식
  3. decrypt(integration.access_token_encrypted) — 표준 v2 OAuth bot token

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

        # mock 판단이 항상 먼저 — SLACK_MCP_REAL=false이면 무조건 mock
        if not self.is_real_mode():
            return self._mock(params, call_id)

        # real mode: tenant OAuth 필수 (SLACK_BOT_TOKEN env fallback 없음)
        if not self._use_tenant_oauth() or not tenant_id:
            logger.warning(
                "SlackConnector: tenant OAuth required call_id=%s tenant_id=%r",
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
                "SlackConnector: token 복호화 실패 call_id=%s tenant_id=%s",
                call_id, tenant_id,
            )
            return self._failed("tenant_token_decryption_failed")

        # metadata에서 bot token 우선 추출 (v1 bot_access_token / v2 raw access_token)
        bot_token = self._extract_bot_token(integration, access_token)
        return await self._post_message(bot_token, params, call_id=call_id)

    # ── .env BOT_TOKEN 실행 (미사용 — 하위 호환 보존용) ──────────────────────────

    async def _execute_env_real(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
    ) -> dict:
        """SLACK_BOT_TOKEN env 기반 실행. execute()에서 호출하지 않는다."""
        bot_token = os.getenv("SLACK_BOT_TOKEN")
        if not bot_token:
            logger.warning("SlackConnector: SLACK_BOT_TOKEN 없음 call_id=%s", call_id)
            return self._skipped("slack_bot_token_not_configured")
        return await self._post_message(bot_token, params, call_id=call_id)

    # ── bot token 추출 헬퍼 ──────────────────────────────────────────────────

    @staticmethod
    def _extract_bot_token(integration, decrypted: str) -> str:
        """integration.metadata에서 Slack bot token을 우선 추출한다.

        우선순위:
          1. metadata["bot"]["bot_access_token"]  — Slack OAuth v1 response
          2. metadata["access_token"] (xoxb- 시작) — v2 raw 저장 시
          3. decrypted                             — access_token_encrypted 복호화 값

        token 원문은 이 함수 내부에서도 로그에 출력하지 않는다.
        """
        meta: dict = getattr(integration, "metadata", None) or {}
        v1_bot = meta.get("bot", {}).get("bot_access_token") or ""
        if v1_bot:
            return v1_bot
        meta_access = meta.get("access_token") or ""
        if meta_access.startswith("xoxb-"):
            return meta_access
        return decrypted

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
