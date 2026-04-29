"""
Gmail MCP Connector.

지원 action_type:
  - send_manager_email

── 실행 정책 ─────────────────────────────────────────────────────────────────
  GMAIL_MCP_REAL=false (기본) → 항상 mock 반환  (MCP_USE_TENANT_OAUTH 값 무관)
  GMAIL_MCP_REAL=true  + MCP_USE_TENANT_OAUTH=false → skipped("tenant_oauth_required")
  GMAIL_MCP_REAL=true  + MCP_USE_TENANT_OAUTH=true + tenant_id 없음 → skipped("tenant_oauth_required")
  GMAIL_MCP_REAL=true  + MCP_USE_TENANT_OAUTH=true + integration 없음 → skipped("tenant_integration_not_connected")
  GMAIL_MCP_REAL=true  + MCP_USE_TENANT_OAUTH=true + integration 있음 → Gmail API 호출

── 수신자 우선순위 ──────────────────────────────────────────────────────────
  params["to"] > GMAIL_MANAGER_TO env

── Gmail API ─────────────────────────────────────────────────────────────────
  POST https://gmail.googleapis.com/gmail/v1/users/me/messages/send
  Authorization: Bearer {tenant_access_token}
  Body: {"raw": "{base64url_mime}"}

── mock mode ─────────────────────────────────────────────────────────────────
  status: success
  external_id: gmail-mock-{call_id}
  result: {sent, to, subject, body_preview, mock}
"""
from __future__ import annotations

import base64
import email.message
import os
from datetime import datetime

from app.services.mcp.connectors.base import BaseMCPConnector
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GmailConnector(BaseMCPConnector):
    connector_name = "gmail"
    _real_mode_env = "GMAIL_MCP_REAL"
    _required_config = ("GMAIL_MANAGER_TO",)
    _oauth_provider_name = "google_gmail"

    async def execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        logger.info(
            "GmailConnector call_id=%s action_type=%s real_mode=%s tenant_oauth=%s",
            call_id, action_type, self.is_real_mode(), self._use_tenant_oauth(),
        )

        # mock 판단이 항상 먼저 — GMAIL_MCP_REAL=false이면 무조건 mock
        if not self.is_real_mode():
            return self._mock(params, call_id)

        # real mode: tenant OAuth 필수
        if not self._use_tenant_oauth() or not tenant_id:
            logger.warning(
                "GmailConnector: tenant OAuth required call_id=%s tenant_id=%r",
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
                "GmailConnector: token 복호화 실패 call_id=%s tenant_id=%s",
                call_id, tenant_id,
            )
            return self._failed("tenant_token_decryption_failed")

        to = params.get("to") or os.getenv("GMAIL_MANAGER_TO", "")
        if not to:
            return self._skipped("gmail_recipient_not_configured")

        subject = params.get("subject") or "[시시콜콜] 상담 후속 조치 알림"
        body = (
            params.get("body")
            or params.get("summary_short")
            or params.get("reason")
            or ""
        )

        return await self._send_email(access_token, to=to, subject=subject, body=body, call_id=call_id)

    # ── Gmail API 호출 ────────────────────────────────────────────────────────

    @staticmethod
    def _build_email_raw(to: str, subject: str, body: str) -> str:
        """MIME 메시지를 생성하고 base64url 인코딩된 raw 문자열을 반환한다."""
        msg = email.message.EmailMessage()
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        return base64.urlsafe_b64encode(msg.as_bytes()).decode().rstrip("=")

    async def _send_email(
        self,
        token: str,
        *,
        to: str,
        subject: str,
        body: str,
        call_id: str,
    ) -> dict:
        """Gmail API messages.send를 호출한다. token은 로그에 출력하지 않는다."""
        import httpx

        raw = self._build_email_raw(to, subject, body)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                    json={"raw": raw},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=15.0,
                )

            if resp.status_code not in (200, 201):
                logger.error(
                    "GmailConnector: HTTP 오류 call_id=%s status=%d",
                    call_id, resp.status_code,
                )
                return self._failed(f"gmail_http_error:{resp.status_code}")

            data = resp.json()
            message_id = data.get("id", "")
            logger.info(
                "GmailConnector: 이메일 전송 완료 call_id=%s message_id=%s to=%s",
                call_id, message_id, to,
            )
            return self._success(
                external_id=message_id,
                result={"message_id": message_id, "to": to, "subject": subject},
            )

        except Exception as exc:
            logger.error(
                "GmailConnector: 예외 call_id=%s err=%s",
                call_id, type(exc).__name__,
            )
            return self._failed(f"gmail_exception:{type(exc).__name__}")

    # ── mock ─────────────────────────────────────────────────────────────────

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
