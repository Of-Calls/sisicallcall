"""
Calendar MCP Connector.

지원 action_type:
  - schedule_callback

── tenant OAuth mode (MCP_USE_TENANT_OAUTH=true) ────────────────────────────
  google_calendar integration token을 사용해 Google Calendar events.insert를 호출.
  tenant_id 기준으로 TenantIntegration을 조회하고 Fernet 복호화한 access_token으로
  Authorization: Bearer 헤더를 구성한다.
  token이 없으면 skipped, API 실패 시 failed, 성공 시 success를 반환한다.
  access_token 원문은 로그에 출력하지 않는다.

── real mode env (.env 계정) ─────────────────────────────────────────────────
  CALENDAR_MCP_REAL=true   real mode 활성화 (현재 미구현 — tenant OAuth 권장)

── mock mode ─────────────────────────────────────────────────────────────────
  status: success
  external_id: calendar-mock-{call_id}
  result: {scheduled, title, preferred_time, customer_phone, reason, mock}

── Calendar API 관련 env ─────────────────────────────────────────────────────
  GOOGLE_CALENDAR_ID          캘린더 ID (기본: primary)
  CALENDAR_DEFAULT_DURATION_MIN 기본 일정 길이 분 (기본: 30)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from app.services.mcp.connectors.base import BaseMCPConnector
from app.utils.logger import get_logger

logger = get_logger(__name__)

_CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3/calendars"
_DEFAULT_TZ = "Asia/Seoul"
_DEFAULT_DURATION_MIN = 30


class CalendarConnector(BaseMCPConnector):
    connector_name = "calendar"
    _real_mode_env = "CALENDAR_MCP_REAL"
    _required_config = ()   # tenant OAuth 모드에서는 env config 불필요
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

        # tenant OAuth 우선 처리
        if self._use_tenant_oauth() and tenant_id:
            return await self._oauth_execute(action_type, params, call_id=call_id, tenant_id=tenant_id)

        if not self.is_real_mode():
            return self._mock(params, call_id)

        # .env real mode — 미구현 (tenant OAuth 사용 권장)
        logger.warning("CalendarConnector: .env real mode 미구현 call_id=%s", call_id)
        return self._skipped("calendar_mcp_real_not_implemented")

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
                return self._mock(params, call_id) if not self.is_real_mode() else self._skipped("calendar_mcp_real_not_implemented")
            return self._skipped("tenant_integration_not_connected")

        # 만료 체크 (naive UTC 비교)
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
                "CalendarConnector: token 복호화 실패 call_id=%s tenant_id=%s",
                call_id, tenant_id,
            )
            return self._failed("tenant_token_decryption_failed")

        return await self._insert_google_event(access_token, params, call_id=call_id)

    async def _insert_google_event(
        self,
        access_token: str,
        params: dict,
        *,
        call_id: str,
    ) -> dict:
        """Google Calendar events.insert API 호출.

        access_token은 로그에 출력하지 않는다.
        """
        import httpx

        calendar_id = (
            params.get("calendar_id")
            or os.getenv("GOOGLE_CALENDAR_ID", "primary")
        )
        url = f"{_CALENDAR_API_BASE}/{calendar_id}/events"
        event_body = self._build_event_body(params)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json=event_body,
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=15.0,
                )

            if resp.status_code not in (200, 201):
                preview = resp.text[:200]
                logger.error(
                    "CalendarConnector: API 오류 call_id=%s status=%d preview=%s",
                    call_id, resp.status_code, preview,
                )
                return self._failed(f"google_calendar_api_error:{resp.status_code}")

            data = resp.json()
            logger.info(
                "CalendarConnector: 일정 생성 완료 call_id=%s event_id=%s",
                call_id, data.get("id"),
            )
            return self._success(
                external_id=data.get("id"),
                result={
                    "event_id": data.get("id"),
                    "html_link": data.get("htmlLink"),
                    "start": (data.get("start") or {}).get("dateTime"),
                    "end": (data.get("end") or {}).get("dateTime"),
                },
            )

        except Exception as exc:
            logger.error(
                "CalendarConnector: events.insert 예외 call_id=%s err=%s",
                call_id, type(exc).__name__,
            )
            return self._failed(f"calendar_insert_exception:{type(exc).__name__}")

    # ── 이벤트 바디 생성 ──────────────────────────────────────────────────────

    def _build_event_body(self, params: dict) -> dict:
        title = params.get("title") or "고객 콜백 일정"

        description = ""
        for key in ("description", "reason", "callback_reason", "summary_short"):
            val = params.get(key)
            if val:
                description = str(val)
                break

        tz = params.get("timezone", _DEFAULT_TZ)
        duration = int(os.getenv("CALENDAR_DEFAULT_DURATION_MIN", str(_DEFAULT_DURATION_MIN)))

        start_str = params.get("start_time") or params.get("preferred_time")
        end_str = params.get("end_time")

        if start_str:
            try:
                start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                start_dt = datetime.utcnow() + timedelta(hours=1)
        else:
            start_dt = datetime.utcnow() + timedelta(hours=1)

        if end_str:
            try:
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                end_dt = start_dt + timedelta(minutes=duration)
        else:
            end_dt = start_dt + timedelta(minutes=duration)

        return {
            "summary": title,
            "description": description,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": tz},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": tz},
        }

    # ── mock ─────────────────────────────────────────────────────────────────

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
