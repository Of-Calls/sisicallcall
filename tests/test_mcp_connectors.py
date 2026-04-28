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
async def test_gmail_connector_mock_success():
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
async def test_calendar_connector_mock_success():
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
async def test_mcp_client_routes_to_correct_connector():
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
