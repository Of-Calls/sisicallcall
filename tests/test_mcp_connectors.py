"""
MCP Connector 계층 테스트.

검증 범위:
  1.  Gmail connector mock success
  2.  Calendar connector mock success
  3.  Jira connector mock success
  4.  Slack connector mock success
  5.  CompanyDB connector mock success
  6.  real mode env 켰지만 설정 부족 → skipped 또는 failed 반환
  7.  MCPClient가 tool_name으로 connector를 찾아 실행
  8.  MCPClient unknown tool → failed 반환
  9.  connector 예외 발생 시 → failed 반환
  10. connector 결과가 status/external_id/result/error 4개 키를 모두 포함
  11. MCPClient.registered_tools()로 등록된 tool 목록 확인
  12. CompanyDB: MCP_COMPANY_DB_REAL env도 real mode로 인식
  13. real mode env + config OK 이어도 connector 미구현이면 skipped
  14. tenant OAuth: 연동된 테넌트 → tenant_token_found_but_real_execute_not_implemented (Gmail)
  15. tenant OAuth: 미연동 테넌트 → tenant_integration_not_connected
  16. tenant OAuth: 연동 후 폴백 없음 → env fallback 없이 skipped 반환
  17. tenant OAuth: MCP_ALLOW_ENV_FALLBACK=true + 미연동 → mock 결과 반환
  18. tenant OAuth: _oauth_provider_name 설정 확인
  19. Calendar: tenant token 있을 때 Google Calendar API events.insert 성공
  20. Calendar: Google Calendar API HTTP 오류 → failed 반환
  21. Calendar: tenant token 없음 → skipped("tenant_integration_not_connected")
  22. Calendar: params calendar_id > GOOGLE_CALENDAR_ID env > primary 우선순위
  23. Calendar: start_time/end_time 직접 지정 시 이벤트 바디에 반영
  24. Calendar: preferred_time만 있을 때 end_time 자동 생성 (default duration)
  25. Calendar: 시간 정보 없을 때 현재+1시간 기본값 사용
  26. Calendar: 결과에 access_token 평문 미포함
"""
from __future__ import annotations

import pytest

from app.services.mcp.connectors.gmail_connector import GmailConnector
from app.services.mcp.connectors.calendar_connector import CalendarConnector
from app.services.mcp.connectors.jira_connector import JiraConnector
from app.services.mcp.connectors.slack_connector import SlackConnector
from app.services.mcp.connectors.company_db_connector import CompanyDBConnector
from app.services.mcp.client import MCPClient


# ── 1. Gmail connector mock success ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_gmail_connector_mock_success(monkeypatch):
    monkeypatch.delenv("GMAIL_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email",
        {"to": "manager@example.com", "subject": "테스트", "body": "본문"},
        call_id="conn-001",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "gmail-mock-conn-001"
    assert result["result"]["sent"] is True
    assert result["result"]["to"] == "manager@example.com"
    assert result["result"]["subject"] == "테스트"
    assert result["result"]["mock"] is True
    assert result["error"] is None


# ── 2. Calendar connector mock success ───────────────────────────────────────

@pytest.mark.asyncio
async def test_calendar_connector_mock_success(monkeypatch):
    monkeypatch.delenv("CALENDAR_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    connector = CalendarConnector()
    result = await connector.execute(
        "schedule_callback",
        {"title": "콜백 예약", "customer_phone": "010-1234-5678"},
        call_id="conn-002",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "calendar-mock-conn-002"
    assert result["result"]["scheduled"] is True
    assert result["result"]["title"] == "콜백 예약"
    assert result["result"]["customer_phone"] == "010-1234-5678"
    assert result["result"]["mock"] is True
    assert result["error"] is None


# ── 3. Jira connector mock success ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_jira_connector_mock_success():
    connector = JiraConnector()
    result = await connector.execute(
        "create_jira_issue",
        {"summary_short": "요금 오류", "reason": "청구서 금액 불일치", "labels": ["billing"]},
        call_id="conn-003",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "jira-mock-conn-003"
    assert result["result"]["mock"] is True
    assert result["result"]["summary"] == "요금 오류"
    assert result["result"]["description"] == "청구서 금액 불일치"
    assert result["result"]["labels"] == ["billing"]
    assert result["error"] is None


# ── 4. Slack connector mock success ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_slack_connector_mock_success():
    connector = SlackConnector()
    result = await connector.execute(
        "send_slack_alert",
        {"channel": "#alerts", "message": "[CRITICAL] 테스트"},
        call_id="conn-004",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "slack-mock-conn-004"
    assert result["result"]["channel"] == "#alerts"
    assert result["result"]["message"] == "[CRITICAL] 테스트"
    assert result["result"]["mock"] is True
    assert result["error"] is None


# ── 5. CompanyDB connector mock success ──────────────────────────────────────

@pytest.mark.asyncio
async def test_company_db_connector_mock_success():
    connector = CompanyDBConnector()
    result = await connector.execute(
        "create_voc_issue",
        {"tier": "high", "priority": "urgent", "primary_category": "billing", "reason": "요금 오류"},
        call_id="conn-005",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "VOC-MOCK-conn-005"
    assert result["result"]["created"] is True
    assert result["result"]["tier"] == "high"
    assert result["result"]["priority"] == "urgent"
    assert result["result"]["primary_category"] == "billing"
    assert result["result"]["mock"] is True
    assert result["error"] is None


# ── 6. real mode env 켰지만 설정 부족 → skipped 반환 ─────────────────────────

@pytest.mark.asyncio
async def test_gmail_connector_real_mode_config_missing_returns_skipped(monkeypatch):
    monkeypatch.setenv("GMAIL_MCP_REAL", "true")
    monkeypatch.delenv("GMAIL_MANAGER_TO", raising=False)

    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email", {}, call_id="conn-006",
    )

    assert result["status"] in ("skipped", "failed")
    assert result["error"] is not None


@pytest.mark.asyncio
async def test_jira_connector_real_mode_config_missing_returns_skipped(monkeypatch):
    monkeypatch.setenv("JIRA_MCP_REAL", "true")
    monkeypatch.delenv("JIRA_PROJECT_KEY", raising=False)
    monkeypatch.delenv("JIRA_ISSUE_TYPE", raising=False)

    connector = JiraConnector()
    result = await connector.execute(
        "create_jira_issue", {}, call_id="conn-006b",
    )

    assert result["status"] in ("skipped", "failed")
    assert result["error"] == "jira_mcp_connector_not_configured"


@pytest.mark.asyncio
async def test_slack_connector_real_mode_config_missing_returns_skipped(monkeypatch):
    monkeypatch.setenv("SLACK_MCP_REAL", "true")
    monkeypatch.delenv("SLACK_ALERT_CHANNEL", raising=False)

    connector = SlackConnector()
    result = await connector.execute(
        "send_slack_alert", {}, call_id="conn-006c",
    )

    assert result["status"] in ("skipped", "failed")
    assert result["error"] == "slack_mcp_connector_not_configured"


# ── 7. MCPClient가 tool_name으로 connector를 찾아 실행 ────────────────────────

@pytest.mark.asyncio
async def test_mcp_client_routes_to_correct_connector(monkeypatch):
    monkeypatch.delenv("GMAIL_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    client = MCPClient()
    client.register_connector("gmail", GmailConnector())

    result = await client.call_tool(
        "gmail", "send_manager_email",
        {"subject": "라우팅 테스트"},
        call_id="client-001",
    )

    assert result["status"] == "success"
    assert "gmail-mock-client-001" == result["external_id"]


# ── 8. MCPClient unknown tool → failed 반환 ──────────────────────────────────

@pytest.mark.asyncio
async def test_mcp_client_unknown_tool_returns_failed():
    client = MCPClient()
    result = await client.call_tool(
        "nonexistent_tool", "some_action", {},
        call_id="client-002",
    )

    assert result["status"] == "failed"
    assert result["error"] is not None
    assert "unknown tool" in result["error"]


# ── 9. connector 예외 발생 시 → failed 반환 ──────────────────────────────────

@pytest.mark.asyncio
async def test_mcp_client_connector_exception_returns_failed():
    from app.services.mcp.connectors.base import BaseMCPConnector

    class ExplodingConnector(BaseMCPConnector):
        connector_name = "exploding"

        async def execute(self, action_type, params, *, call_id, tenant_id=""):
            raise RuntimeError("의도적 폭발")

    client = MCPClient()
    client.register_connector("exploding", ExplodingConnector())

    result = await client.call_tool(
        "exploding", "boom", {},
        call_id="client-003",
    )

    assert result["status"] == "failed"
    assert "의도적 폭발" in result["error"]


# ── 10. connector 결과가 4개 표준 키를 포함 ──────────────────────────────────

@pytest.mark.asyncio
async def test_connector_result_has_standard_keys():
    connectors = [
        ("gmail", GmailConnector(), "send_manager_email", {}),
        ("calendar", CalendarConnector(), "schedule_callback", {}),
        ("jira", JiraConnector(), "create_jira_issue", {}),
        ("slack", SlackConnector(), "send_slack_alert", {}),
        ("company_db", CompanyDBConnector(), "create_voc_issue", {}),
    ]
    for name, connector, action_type, params in connectors:
        result = await connector.execute(action_type, params, call_id=f"key-test-{name}")
        for key in ("status", "external_id", "result", "error"):
            assert key in result, f"{name} connector 결과에 {key!r} 키가 없음"


# ── 11. MCPClient.registered_tools() ─────────────────────────────────────────

def test_mcp_client_registered_tools():
    from app.services.mcp.client import mcp_client

    tools = mcp_client.registered_tools()
    for expected in ("gmail", "calendar", "company_db", "jira", "slack"):
        assert expected in tools, f"기본 등록 tool {expected!r} 이 없음"


# ── 12. CompanyDB: MCP_COMPANY_DB_REAL도 real mode로 인식 ─────────────────────

def test_company_db_connector_legacy_env_var(monkeypatch):
    monkeypatch.setenv("MCP_COMPANY_DB_REAL", "true")
    monkeypatch.delenv("COMPANY_DB_MCP_REAL", raising=False)

    connector = CompanyDBConnector()
    assert connector.is_real_mode() is True


# ── 13. real mode + config OK → connector 미구현이면 skipped ─────────────────

@pytest.mark.asyncio
async def test_calendar_connector_real_mode_config_ok_returns_skipped(monkeypatch):
    monkeypatch.setenv("CALENDAR_MCP_REAL", "true")
    monkeypatch.setenv("CALENDAR_DEFAULT_OWNER", "owner@example.com")

    connector = CalendarConnector()
    assert connector.is_real_mode() is True
    ok, err = connector.validate_config()
    assert ok is True

    result = await connector.execute(
        "schedule_callback", {}, call_id="conn-013",
    )

    assert result["status"] in ("skipped", "failed")


# ── 14. tenant OAuth: 연동된 테넌트 → not_implemented skipped ────────────────

@pytest.mark.asyncio
async def test_gmail_connector_tenant_oauth_connected(monkeypatch):
    from cryptography.fernet import Fernet
    from app.models.tenant_integration import TenantIntegration, IntegrationStatus
    from app.repositories.tenant_integration_repo import (
        tenant_integration_repo, upsert_integration,
    )
    from app.services.oauth.token_crypto import reset_fernet_cache

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    reset_fernet_cache()
    tenant_integration_repo.clear_integrations()

    from app.services.oauth.token_crypto import encrypt_token
    enc_token = encrypt_token("fake-access-token")

    upsert_integration(TenantIntegration(
        tenant_id="tenant-oauth-001",
        provider="google_gmail",
        status=IntegrationStatus.connected,
        access_token_encrypted=enc_token,
    ))

    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email", {},
        call_id="oauth-001",
        tenant_id="tenant-oauth-001",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "tenant_token_found_but_real_execute_not_implemented"

    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── 15. tenant OAuth: 미연동 테넌트 → not_connected skipped ─────────────────

@pytest.mark.asyncio
async def test_gmail_connector_tenant_oauth_not_connected(monkeypatch):
    from app.repositories.tenant_integration_repo import tenant_integration_repo

    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    tenant_integration_repo.clear_integrations()

    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email", {},
        call_id="oauth-002",
        tenant_id="no-such-tenant",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "tenant_integration_not_connected"

    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)


# ── 16. tenant OAuth: MCP_ALLOW_ENV_FALLBACK 없음 → not_connected 반환 ───────

@pytest.mark.asyncio
async def test_gmail_connector_tenant_oauth_no_fallback(monkeypatch):
    from app.repositories.tenant_integration_repo import tenant_integration_repo

    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.delenv("MCP_ALLOW_ENV_FALLBACK", raising=False)
    tenant_integration_repo.clear_integrations()

    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email", {},
        call_id="oauth-003",
        tenant_id="no-such-tenant",
    )

    # fallback 없으므로 not_connected skipped 반환 (mock으로 내려가지 않음)
    assert result["status"] == "skipped"
    assert result["error"] == "tenant_integration_not_connected"

    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)


# ── 17. tenant OAuth: MCP_ALLOW_ENV_FALLBACK=true + 미연동 → mock 반환 ───────

@pytest.mark.asyncio
async def test_gmail_connector_tenant_oauth_with_env_fallback(monkeypatch):
    from app.repositories.tenant_integration_repo import tenant_integration_repo

    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("MCP_ALLOW_ENV_FALLBACK", "true")
    monkeypatch.delenv("GMAIL_MCP_REAL", raising=False)
    tenant_integration_repo.clear_integrations()

    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email",
        {"to": "manager@example.com", "subject": "fallback test"},
        call_id="oauth-004",
        tenant_id="no-such-tenant",
    )

    # 미연동 + fallback 허용 → mock 결과
    assert result["status"] == "success"
    assert result["result"]["mock"] is True

    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("MCP_ALLOW_ENV_FALLBACK", raising=False)


# ── 18. tenant OAuth: _oauth_provider_name 설정 확인 ─────────────────────────

def test_connectors_have_oauth_provider_name():
    assert GmailConnector._oauth_provider_name == "google_gmail"
    assert CalendarConnector._oauth_provider_name == "google_calendar"
    assert SlackConnector._oauth_provider_name == "slack"
    assert JiraConnector._oauth_provider_name == "jira"
    assert CompanyDBConnector._oauth_provider_name == ""  # OAuth 불필요


# ── Calendar 실제 API 테스트 공통 fixture ─────────────────────────────────────

def _setup_calendar_tenant(monkeypatch, enc_token: str, tenant_id: str = "cal-tenant") -> None:
    from app.models.tenant_integration import TenantIntegration, IntegrationStatus
    from app.repositories.tenant_integration_repo import tenant_integration_repo, upsert_integration
    tenant_integration_repo.clear_integrations()
    upsert_integration(TenantIntegration(
        tenant_id=tenant_id,
        provider="google_calendar",
        status=IntegrationStatus.connected,
        access_token_encrypted=enc_token,
    ))


def _make_mock_http_client(status_code: int, json_data: dict):
    """httpx.AsyncClient를 대체하는 동기-호환 mock 팩토리."""
    from unittest.mock import MagicMock, AsyncMock

    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.text = str(json_data)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ── 19. Calendar: tenant token → events.insert 성공 ──────────────────────────

@pytest.mark.asyncio
async def test_calendar_real_api_success(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    reset_fernet_cache()

    enc = encrypt_token("ya29.fake_access_token")
    _setup_calendar_tenant(monkeypatch, enc)

    api_response = {
        "id": "event-id-abc123",
        "htmlLink": "https://calendar.google.com/event/abc123",
        "start": {"dateTime": "2026-04-29T10:00:00+09:00"},
        "end": {"dateTime": "2026-04-29T10:30:00+09:00"},
    }
    mock_client = _make_mock_http_client(200, api_response)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = CalendarConnector()
    result = await connector.execute(
        "schedule_callback",
        {"title": "고객 콜백", "reason": "요금 문의 후속"},
        call_id="cal-001",
        tenant_id="cal-tenant",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "event-id-abc123"
    assert result["result"]["event_id"] == "event-id-abc123"
    assert result["result"]["html_link"] == "https://calendar.google.com/event/abc123"
    assert result["error"] is None

    # 4개 표준 키 확인
    for key_ in ("status", "external_id", "result", "error"):
        assert key_ in result

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── 20. Calendar: Google Calendar API HTTP 오류 → failed ─────────────────────

@pytest.mark.asyncio
async def test_calendar_real_api_http_failure(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    reset_fernet_cache()

    enc = encrypt_token("ya29.fake_access_token")
    _setup_calendar_tenant(monkeypatch, enc)

    mock_client = _make_mock_http_client(401, {"error": "Unauthorized"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = CalendarConnector()
    result = await connector.execute(
        "schedule_callback", {},
        call_id="cal-002",
        tenant_id="cal-tenant",
    )

    assert result["status"] == "failed"
    assert "401" in result["error"]

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── 21. Calendar: tenant token 없음 → skipped ────────────────────────────────

@pytest.mark.asyncio
async def test_calendar_no_tenant_integration_skipped(monkeypatch):
    from app.repositories.tenant_integration_repo import tenant_integration_repo

    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.delenv("MCP_ALLOW_ENV_FALLBACK", raising=False)
    tenant_integration_repo.clear_integrations()

    connector = CalendarConnector()
    result = await connector.execute(
        "schedule_callback", {},
        call_id="cal-003",
        tenant_id="no-calendar-tenant",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "tenant_integration_not_connected"

    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)


# ── 22. Calendar: calendar_id 우선순위 (params > env > primary) ───────────────

@pytest.mark.asyncio
async def test_calendar_calendar_id_priority(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token
    from unittest.mock import AsyncMock, MagicMock

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("GOOGLE_CALENDAR_ID", "env-calendar@group.calendar.google.com")
    reset_fernet_cache()

    enc = encrypt_token("ya29.fake_token")
    _setup_calendar_tenant(monkeypatch, enc)

    captured_urls: list[str] = []

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": "ev1", "htmlLink": "", "start": {}, "end": {}}
    mock_resp.text = ""

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def _post(url, **kwargs):
        captured_urls.append(url)
        return mock_resp

    mock_client.post = _post
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = CalendarConnector()

    # params calendar_id 우선
    captured_urls.clear()
    await connector.execute(
        "schedule_callback",
        {"calendar_id": "params-calendar@group.calendar.google.com"},
        call_id="cal-004a", tenant_id="cal-tenant",
    )
    assert "params-calendar" in captured_urls[0]

    # env GOOGLE_CALENDAR_ID (params 없음)
    captured_urls.clear()
    await connector.execute(
        "schedule_callback", {},
        call_id="cal-004b", tenant_id="cal-tenant",
    )
    assert "env-calendar" in captured_urls[0]

    # primary (params 없음, env 없음)
    monkeypatch.delenv("GOOGLE_CALENDAR_ID", raising=False)
    captured_urls.clear()
    await connector.execute(
        "schedule_callback", {},
        call_id="cal-004c", tenant_id="cal-tenant",
    )
    assert "/primary/" in captured_urls[0]

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── 23. Calendar: start_time/end_time 직접 지정 ──────────────────────────────

def test_calendar_event_body_explicit_start_end():
    connector = CalendarConnector()
    body = connector._build_event_body({
        "title": "명시 시간 테스트",
        "start_time": "2026-05-01T09:00:00",
        "end_time": "2026-05-01T10:00:00",
    })

    assert body["summary"] == "명시 시간 테스트"
    assert "2026-05-01T09:00:00" in body["start"]["dateTime"]
    assert "2026-05-01T10:00:00" in body["end"]["dateTime"]


# ── 24. Calendar: preferred_time → end_time 자동 생성 ────────────────────────

def test_calendar_event_body_preferred_time_auto_end():
    connector = CalendarConnector()
    body = connector._build_event_body({
        "preferred_time": "2026-05-01T14:00:00",
    })

    from datetime import datetime, timedelta
    start_dt = datetime.fromisoformat(body["start"]["dateTime"])
    end_dt = datetime.fromisoformat(body["end"]["dateTime"])

    assert start_dt.hour == 14
    diff = end_dt - start_dt
    assert diff == timedelta(minutes=30)  # CALENDAR_DEFAULT_DURATION_MIN 기본값


# ── 25. Calendar: 시간 없을 때 현재+1시간 ────────────────────────────────────

def test_calendar_event_body_default_time():
    from datetime import datetime, timedelta

    before = datetime.utcnow() + timedelta(minutes=55)  # 약간의 여유
    connector = CalendarConnector()
    body = connector._build_event_body({"title": "시간 없음"})
    after = datetime.utcnow() + timedelta(hours=1, minutes=5)

    start_dt = datetime.fromisoformat(body["start"]["dateTime"])
    assert before <= start_dt <= after


# ── 26. Calendar: 결과에 access_token 평문 미포함 ────────────────────────────

@pytest.mark.asyncio
async def test_calendar_access_token_not_in_result(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token

    plaintext_token = "ya29.super_secret_access_token_do_not_leak"

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    reset_fernet_cache()

    enc = encrypt_token(plaintext_token)
    _setup_calendar_tenant(monkeypatch, enc)

    api_response = {"id": "ev-safe", "htmlLink": "", "start": {}, "end": {}}
    mock_client = _make_mock_http_client(200, api_response)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = CalendarConnector()
    result = await connector.execute(
        "schedule_callback", {},
        call_id="cal-safe",
        tenant_id="cal-tenant",
    )

    # result dict를 str로 변환해도 평문 토큰이 없어야 함
    result_str = str(result)
    assert plaintext_token not in result_str
    assert result["status"] == "success"

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)
