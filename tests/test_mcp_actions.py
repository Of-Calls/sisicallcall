import pytest
from app.agents.post_call.actions.executor import ActionExecutor, execute_actions
from app.agents.post_call.actions.gmail_action import GmailAction
from app.agents.post_call.actions.company_db_action import CompanyDBAction
from app.agents.post_call.actions.calendar_action import CalendarAction
from app.agents.post_call.actions.result import action_success, action_failed, action_skipped
from app.agents.post_call.actions.registry import (
    get_handler, register, unregister, registered_tools,
)
from app.agents.post_call.schemas import ActionType, Tool, ActionStatus


@pytest.fixture
def executor():
    return ActionExecutor()


# ── 기존 executor 라우팅 테스트 ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_executor_gmail_action(executor):
    action = {
        "action_type": ActionType.send_manager_email.value,
        "tool": Tool.gmail.value,
        "params": {"subject": "테스트 메일", "to": "test@example.com"},
        "status": ActionStatus.pending.value,
    }
    results = await executor.execute_all([action], call_id="t-001")

    assert len(results) == 1
    assert results[0]["status"] == "success"
    assert results[0]["result"]["sent"] is True


@pytest.mark.asyncio
async def test_executor_company_db_action(executor):
    action = {
        "action_type": ActionType.create_voc_issue.value,
        "tool": Tool.company_db.value,
        "params": {"tier": "high"},
        "status": ActionStatus.pending.value,
    }
    results = await executor.execute_all([action], call_id="t-002")

    assert results[0]["status"] == "success"
    assert "issue_id" in results[0]["result"]
    assert results[0]["result"]["tier"] == "high"


@pytest.mark.asyncio
async def test_executor_calendar_action(executor):
    action = {
        "action_type": ActionType.schedule_callback.value,
        "tool": Tool.calendar.value,
        "params": {"title": "재콜백 예약"},
        "status": ActionStatus.pending.value,
    }
    results = await executor.execute_all([action], call_id="t-003")

    assert results[0]["status"] == "success"
    assert results[0]["result"]["scheduled"] is True
    assert results[0]["result"]["title"] == "재콜백 예약"


@pytest.mark.asyncio
async def test_executor_internal_dashboard_action(executor):
    action = {
        "action_type": ActionType.add_priority_queue.value,
        "tool": Tool.internal_dashboard.value,
        "params": {},
        "status": ActionStatus.pending.value,
    }
    results = await executor.execute_all([action], call_id="t-004")
    assert results[0]["status"] == "success"


@pytest.mark.asyncio
async def test_executor_unknown_tool_failed(executor):
    """알 수 없는 tool 은 skipped 가 아니라 failed 여야 한다."""
    action = {
        "action_type": "unknown_action",
        "tool": "unknown_tool",
        "params": {},
        "status": "pending",
    }
    results = await executor.execute_all([action], call_id="t-005")
    assert results[0]["status"] == "failed"
    assert "unknown tool" in results[0]["error"]


@pytest.mark.asyncio
async def test_executor_multiple_actions(executor):
    actions = [
        {
            "action_type": ActionType.create_voc_issue.value,
            "tool": Tool.company_db.value,
            "params": {"tier": "medium"},
            "status": ActionStatus.pending.value,
        },
        {
            "action_type": ActionType.send_manager_email.value,
            "tool": Tool.gmail.value,
            "params": {"subject": "복수 액션 테스트"},
            "status": ActionStatus.pending.value,
        },
    ]
    results = await executor.execute_all(actions, call_id="t-006")

    assert len(results) == 2
    assert all(r["status"] == "success" for r in results)


# ── 핸들러 직접 호출 테스트 (새 포맷) ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_gmail_action_direct():
    handler = GmailAction()
    result = await handler.execute(
        {"action_type": "send_manager_email", "params": {"subject": "직접 호출"}},
        call_id="direct-001",
    )
    assert result["status"] == "success"
    assert result["result"]["sent"] is True
    assert result["result"]["mock"] is True
    assert result["external_id"] == "gmail-mock-direct-001"


@pytest.mark.asyncio
async def test_company_db_action_direct():
    handler = CompanyDBAction()
    result = await handler.execute(
        {"action_type": "create_voc_issue", "params": {"tier": "critical"}},
        call_id="direct-002",
    )
    assert result["status"] == "success"
    assert result["result"]["created"] is True
    assert result["result"]["tier"] == "critical"
    assert "issue_id" in result["result"]
    assert result["external_id"] == "VOC-MOCK-direct-002"


@pytest.mark.asyncio
async def test_calendar_action_direct():
    handler = CalendarAction()
    result = await handler.execute(
        {"action_type": "schedule_callback", "params": {"title": "직접 재콜"}},
        call_id="direct-003",
    )
    assert result["status"] == "success"
    assert result["result"]["scheduled"] is True
    assert result["result"]["title"] == "직접 재콜"
    assert result["external_id"] == "calendar-mock-direct-003"


@pytest.mark.asyncio
async def test_executor_preserves_original_action_fields(executor):
    action = {
        "action_type": ActionType.mark_faq_candidate.value,
        "tool": Tool.internal_dashboard.value,
        "params": {"text": "요금제 변경 방법"},
        "status": ActionStatus.pending.value,
    }
    results = await executor.execute_all([action], call_id="t-007")
    assert results[0]["action_type"] == ActionType.mark_faq_candidate.value
    assert results[0]["params"]["text"] == "요금제 변경 방법"


# ── execute_actions 인터페이스 테스트 ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_actions_module_level_gmail():
    actions = [{
        "action_type": ActionType.send_manager_email.value,
        "tool": Tool.gmail.value,
        "params": {"to": "a@b.com", "subject": "모듈 레벨 테스트"},
        "status": ActionStatus.pending.value,
    }]
    results = await execute_actions(call_id="ea-001", tenant_id="tenant-x", actions=actions)
    assert len(results) == 1
    assert results[0]["status"] == "success"
    assert results[0]["result"]["sent"] is True


@pytest.mark.asyncio
async def test_execute_actions_none_returns_empty():
    results = await execute_actions(call_id="ea-002", tenant_id="t", actions=None)
    assert results == []


@pytest.mark.asyncio
async def test_execute_actions_empty_list_returns_empty():
    results = await execute_actions(call_id="ea-003", tenant_id="t", actions=[])
    assert results == []


@pytest.mark.asyncio
async def test_execute_actions_standard_6_key_format():
    actions = [{
        "action_type": ActionType.create_voc_issue.value,
        "tool": Tool.company_db.value,
        "params": {"tier": "low"},
        "status": ActionStatus.pending.value,
    }]
    results = await execute_actions(call_id="ea-004", tenant_id="t", actions=actions)
    r = results[0]
    for key in ("action_type", "tool", "status", "external_id", "error", "result"):
        assert key in r, f"결과에 {key!r} 키가 없음"


@pytest.mark.asyncio
async def test_execute_actions_external_id_format_gmail():
    actions = [{
        "action_type": ActionType.send_manager_email.value,
        "tool": Tool.gmail.value,
        "params": {},
        "status": ActionStatus.pending.value,
    }]
    results = await execute_actions(call_id="ea-005", tenant_id="t", actions=actions)
    assert results[0]["external_id"] == "gmail-mock-ea-005"


@pytest.mark.asyncio
async def test_execute_actions_external_id_format_company_db():
    actions = [{
        "action_type": ActionType.create_voc_issue.value,
        "tool": Tool.company_db.value,
        "params": {},
        "status": ActionStatus.pending.value,
    }]
    results = await execute_actions(call_id="ea-006", tenant_id="t", actions=actions)
    assert results[0]["external_id"] == "VOC-MOCK-ea-006"


@pytest.mark.asyncio
async def test_execute_actions_external_id_format_calendar():
    actions = [{
        "action_type": ActionType.schedule_callback.value,
        "tool": Tool.calendar.value,
        "params": {},
        "status": ActionStatus.pending.value,
    }]
    results = await execute_actions(call_id="ea-007", tenant_id="t", actions=actions)
    assert results[0]["external_id"] == "calendar-mock-ea-007"


@pytest.mark.asyncio
async def test_execute_actions_company_db_params_in_result():
    actions = [{
        "action_type": ActionType.create_voc_issue.value,
        "tool": Tool.company_db.value,
        "params": {
            "tier": "high",
            "priority": "urgent",
            "primary_category": "billing",
            "reason": "요금 오류",
            "summary_short": "청구서 금액 불일치",
        },
        "status": ActionStatus.pending.value,
    }]
    results = await execute_actions(call_id="ea-008", tenant_id="t", actions=actions)
    r = results[0]["result"]
    assert r["priority"] == "urgent"
    assert r["primary_category"] == "billing"
    assert r["reason"] == "요금 오류"
    assert r["summary_short"] == "청구서 금액 불일치"


@pytest.mark.asyncio
async def test_execute_actions_unknown_tool_is_failed_not_skipped():
    actions = [{
        "action_type": "noop",
        "tool": "nonexistent_tool",
        "params": {},
        "status": "pending",
    }]
    results = await execute_actions(call_id="ea-009", tenant_id="t", actions=actions)
    assert results[0]["status"] == "failed"
    assert results[0]["error"] is not None


@pytest.mark.asyncio
async def test_execute_actions_one_fail_does_not_stop_others():
    actions = [
        {
            "action_type": "noop",
            "tool": "bad_tool",
            "params": {},
            "status": "pending",
        },
        {
            "action_type": ActionType.schedule_callback.value,
            "tool": Tool.calendar.value,
            "params": {"title": "계속 실행"},
            "status": ActionStatus.pending.value,
        },
    ]
    results = await execute_actions(call_id="ea-010", tenant_id="t", actions=actions)
    assert len(results) == 2
    assert results[0]["status"] == "failed"
    assert results[1]["status"] == "success"


# ── KDT-76 후속 보강: registry / result helper 테스트 ─────────────────────────

def test_default_tools_registered():
    """기본 4개 tool이 registry에 등록되어 있어야 한다."""
    tools = registered_tools()
    for tool in ("gmail", "company_db", "calendar", "internal_dashboard"):
        assert tool in tools, f"기본 tool {tool!r} 이 registry에 없음"


def test_get_handler_returns_handler_for_known_tools():
    """알려진 tool 이름으로 handler를 조회할 수 있어야 한다."""
    for tool in ("gmail", "company_db", "calendar", "internal_dashboard"):
        handler = get_handler(tool)
        assert handler is not None, f"{tool!r} handler가 None"
        assert hasattr(handler, "execute"), f"{tool!r} handler에 execute 메서드 없음"


def test_result_helpers_return_6_standard_keys():
    """action_success/action_failed/action_skipped 가 6개 표준 키를 모두 반환한다."""
    action = {
        "action_type": "test_action",
        "tool": "test_tool",
        "params": {},
        "status": "pending",
    }
    for result in [
        action_success(action),
        action_failed(action, error="테스트 오류"),
        action_skipped(action, reason="조건 미충족"),
    ]:
        for key in ("action_type", "tool", "status", "external_id", "error", "result"):
            assert key in result, f"결과에 표준 키 {key!r} 가 없음"


def test_result_helpers_status_values():
    """각 helper가 올바른 status 값을 반환한다."""
    action = {"action_type": "a", "tool": "t", "params": {}, "status": "pending"}
    assert action_success(action)["status"] == "success"
    assert action_failed(action, error="e")["status"] == "failed"
    assert action_skipped(action, reason="r")["status"] == "skipped"


@pytest.mark.asyncio
async def test_handler_exception_continues_next_action():
    """handler 예외 발생 시 failed 반환 후 다음 action을 계속 실행한다."""

    class ExplodingHandler:
        async def execute(self, action, *, call_id, tenant_id=""):
            raise RuntimeError("의도적 폭발")

    register("exploding_tool", ExplodingHandler())
    try:
        actions = [
            {"action_type": "boom", "tool": "exploding_tool", "params": {}, "status": "pending"},
            {
                "action_type": ActionType.schedule_callback.value,
                "tool": Tool.calendar.value,
                "params": {"title": "after boom"},
                "status": ActionStatus.pending.value,
            },
        ]
        results = await execute_actions(call_id="exc-001", tenant_id="t", actions=actions)
        assert results[0]["status"] == "failed"
        assert "의도적 폭발" in results[0]["error"]
        assert results[1]["status"] == "success"
    finally:
        unregister("exploding_tool")


@pytest.mark.asyncio
async def test_register_new_handler_and_execute():
    """새 dummy handler를 registry에 등록하면 execute_actions로 즉시 실행 가능하다."""

    class DummySlackAction:
        async def execute(self, action, *, call_id, tenant_id=""):
            return {
                "external_id": f"slack-{call_id}",
                "status": "success",
                "result": {"posted": True, "channel": action.get("params", {}).get("channel", "#general")},
            }

    register("slack", DummySlackAction())
    try:
        results = await execute_actions(
            call_id="reg-001",
            tenant_id="t",
            actions=[{
                "action_type": "post_slack_message",
                "tool": "slack",
                "params": {"channel": "#alerts"},
                "status": "pending",
            }],
        )
        assert results[0]["status"] == "success"
        assert results[0]["result"]["posted"] is True
        assert results[0]["result"]["channel"] == "#alerts"
        assert results[0]["external_id"] == "slack-reg-001"
    finally:
        unregister("slack")


@pytest.mark.asyncio
async def test_action_order_preserved():
    """입력 actions 순서와 반환 results 순서가 동일해야 한다."""
    actions = [
        {
            "action_type": ActionType.create_voc_issue.value,
            "tool": Tool.company_db.value,
            "params": {},
            "status": ActionStatus.pending.value,
        },
        {
            "action_type": ActionType.send_manager_email.value,
            "tool": Tool.gmail.value,
            "params": {},
            "status": ActionStatus.pending.value,
        },
        {
            "action_type": ActionType.schedule_callback.value,
            "tool": Tool.calendar.value,
            "params": {},
            "status": ActionStatus.pending.value,
        },
    ]
    results = await execute_actions("order-001", "t", actions)
    assert results[0]["action_type"] == ActionType.create_voc_issue.value
    assert results[1]["action_type"] == ActionType.send_manager_email.value
    assert results[2]["action_type"] == ActionType.schedule_callback.value


def test_real_mode_env_does_not_break_import(monkeypatch):
    """real mode 환경 변수가 켜져도 import 시점에 오류가 발생하지 않아야 한다."""
    monkeypatch.setenv("MCP_GMAIL_REAL", "1")
    monkeypatch.setenv("MCP_COMPANY_DB_REAL", "1")
    monkeypatch.setenv("MCP_CALENDAR_REAL", "1")

    try:
        import app.services.mcp.gmail as gmail_mod
        import app.services.mcp.company_db as cdb_mod
        import app.services.mcp.calendar as cal_mod
        import app.agents.post_call.actions.gmail_action as ga_mod
        import app.agents.post_call.actions.company_db_action as ca_mod
        import app.agents.post_call.actions.calendar_action as cal_a_mod
    except Exception as exc:
        pytest.fail(f"real mode env 설정 시 import 실패: {exc}")

    assert gmail_mod.GmailMCPService is not None
    assert cdb_mod.CompanyDBMCPService is not None
    assert cal_mod.CalendarMCPService is not None
