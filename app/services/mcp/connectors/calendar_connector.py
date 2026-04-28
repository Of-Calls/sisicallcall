"""
Calendar MCP Connector.

지원 action_type:
  - schedule_callback

── real mode env ─────────────────────────────────────────────────────────────
  CALENDAR_MCP_REAL=true         real mode 활성화
  CALENDAR_DEFAULT_OWNER         기본 캘린더 소유자 (필수)
  CALENDAR_DEFAULT_DURATION_MIN  기본 일정 길이 (선택, 기본 30)
  CALENDAR_MCP_SERVER_URL        MCP 서버 URL (선택)

── mock mode ─────────────────────────────────────────────────────────────────
  status: success
  external_id: calendar-mock-{call_id}
  result: {scheduled, title, preferred_time, customer_phone, reason, mock}

── real mode 설정 부족 ────────────────────────────────────────────────────────
  status: skipped
  error: "calendar_mcp_connector_not_configured"
"""
from __future__ import annotations

import os

from app.services.mcp.connectors.base import BaseMCPConnector
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CalendarConnector(BaseMCPConnector):
    connector_name = "calendar"
    _real_mode_env = "CALENDAR_MCP_REAL"
    _required_config = ("CALENDAR_DEFAULT_OWNER",)
    _oauth_provider_name = "google_calendar"

    async def execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        logger.info(
            "CalendarConnector call_id=%s action_type=%s real_mode=%s tenant_oauth=%s",
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
            logger.warning("CalendarConnector: config 부족 call_id=%s err=%s", call_id, err)
            return self._skipped("calendar_mcp_connector_not_configured")

        return await self._execute_real(action_type, params, call_id=call_id)

    def _mock(self, params: dict, call_id: str) -> dict:
        external_id = f"calendar-mock-{call_id}"
        return self._success(
            external_id=external_id,
            result={
                "scheduled": True,
                "event_id": external_id,
                "title": params.get("title", "콜백 예약"),
                "preferred_time": params.get("preferred_time"),
                "customer_phone": params.get("customer_phone"),
                "reason": params.get("callback_reason", ""),
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
        # TODO: 실제 Google Calendar MCP 서버 연동 구현
        # owner = os.getenv("CALENDAR_DEFAULT_OWNER")
        # duration = int(os.getenv("CALENDAR_DEFAULT_DURATION_MIN", "30"))
        # server_url = os.getenv("CALENDAR_MCP_SERVER_URL")
        logger.warning(
            "CalendarConnector: real mode TODO — skipped call_id=%s action_type=%s",
            call_id, action_type,
        )
        return self._skipped("calendar_mcp_real_not_implemented")
