"""
Gmail MCP Connector.

지원 action_type:
  - send_manager_email

── real mode env ─────────────────────────────────────────────────────────────
  GMAIL_MCP_REAL=true        real mode 활성화
  GMAIL_MANAGER_TO           수신자 이메일 (필수)
  GMAIL_MCP_SERVER_URL       MCP 서버 URL (선택)

── mock mode ─────────────────────────────────────────────────────────────────
  status: success
  external_id: gmail-mock-{call_id}
  result: {sent, to, subject, body_preview, mock}

── real mode 설정 부족 ────────────────────────────────────────────────────────
  status: skipped
  error: "gmail_mcp_connector_not_configured"
"""
from __future__ import annotations

import os

from app.services.mcp.connectors.base import BaseMCPConnector
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GmailConnector(BaseMCPConnector):
    connector_name = "gmail"
    _real_mode_env = "GMAIL_MCP_REAL"
    _required_config = ("GMAIL_MANAGER_TO",)

    async def execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        logger.info(
            "GmailConnector call_id=%s action_type=%s real_mode=%s",
            call_id, action_type, self.is_real_mode(),
        )

        if not self.is_real_mode():
            return self._mock(params, call_id)

        ok, err = self.validate_config()
        if not ok:
            logger.warning("GmailConnector: config 부족 call_id=%s err=%s", call_id, err)
            return self._skipped("gmail_mcp_connector_not_configured")

        return await self._execute_real(action_type, params, call_id=call_id)

    def _mock(self, params: dict, call_id: str) -> dict:
        to = params.get("to") or os.getenv("GMAIL_MANAGER_TO", "manager@example.com")
        subject = params.get("subject", "")
        body = params.get("body", "")
        return self._success(
            external_id=f"gmail-mock-{call_id}",
            result={
                "sent": True,
                "to": to,
                "subject": subject,
                "body_preview": body[:100] if body else "",
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
        # TODO: 실제 Gmail MCP 서버 연동 구현
        # server_url = os.getenv("GMAIL_MCP_SERVER_URL")
        # manager_to = os.getenv("GMAIL_MANAGER_TO")
        # async with httpx.AsyncClient() as client:
        #     resp = await client.post(f"{server_url}/send", json={...})
        #     resp.raise_for_status()
        #     data = resp.json()
        #     return self._success(data.get("message_id"), {"sent": True, ...})
        logger.warning(
            "GmailConnector: real mode TODO — skipped call_id=%s action_type=%s",
            call_id, action_type,
        )
        return self._skipped("gmail_mcp_real_not_implemented")
