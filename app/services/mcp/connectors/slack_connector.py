"""
Slack MCP Connector.

지원 action_type:
  - send_slack_alert

── real mode env ─────────────────────────────────────────────────────────────
  SLACK_MCP_REAL=true         real mode 활성화
  SLACK_ALERT_CHANNEL         알림 채널 (필수, 예: #alerts)
  SLACK_CRITICAL_CHANNEL      critical 전용 채널 (선택)
  SLACK_BOT_TOKEN             봇 토큰 (선택)
  SLACK_MCP_SERVER_URL        MCP 서버 URL (선택)

── mock mode ─────────────────────────────────────────────────────────────────
  status: success
  external_id: slack-mock-{call_id}
  result: {channel, message, mock}

── real mode 설정 부족 ────────────────────────────────────────────────────────
  status: skipped
  error: "slack_mcp_connector_not_configured"
"""
from __future__ import annotations

import os

from app.services.mcp.connectors.base import BaseMCPConnector
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SlackConnector(BaseMCPConnector):
    connector_name = "slack"
    _real_mode_env = "SLACK_MCP_REAL"
    _required_config = ("SLACK_ALERT_CHANNEL",)

    async def execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        logger.info(
            "SlackConnector call_id=%s action_type=%s real_mode=%s",
            call_id, action_type, self.is_real_mode(),
        )

        if not self.is_real_mode():
            return self._mock(params, call_id)

        ok, err = self.validate_config()
        if not ok:
            logger.warning("SlackConnector: config 부족 call_id=%s err=%s", call_id, err)
            return self._skipped("slack_mcp_connector_not_configured")

        return await self._execute_real(action_type, params, call_id=call_id)

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

    async def _execute_real(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
    ) -> dict:
        # TODO: 실제 Slack API / MCP 서버 연동 구현
        # channel = params.get("channel") or os.getenv("SLACK_ALERT_CHANNEL")
        # token = os.getenv("SLACK_BOT_TOKEN")
        # server_url = os.getenv("SLACK_MCP_SERVER_URL")
        logger.warning(
            "SlackConnector: real mode TODO — skipped call_id=%s action_type=%s",
            call_id, action_type,
        )
        return self._skipped("slack_mcp_real_not_implemented")
