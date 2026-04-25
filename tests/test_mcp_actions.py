import pytest
from app.agents.post_call.actions.executor import ActionExecutor
from app.agents.post_call.actions.gmail_action import GmailAction
from app.agents.post_call.actions.company_db_action import CompanyDBAction
from app.agents.post_call.actions.calendar_action import CalendarAction
from app.agents.post_call.schemas import ActionType, Tool, ActionStatus


@pytest.fixture
def executor():
    return ActionExecutor()


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
async def test_executor_unknown_tool_skipped(executor):
    action = {
        "action_type": "unknown_action",
        "tool": "unknown_tool",
        "params": {},
        "status": "pending",
    }
    results = await executor.execute_all([action], call_id="t-005")
    assert results[0]["status"] == "skipped"
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


@pytest.mark.asyncio
async def test_gmail_action_direct():
    handler = GmailAction()
    result = await handler.execute(
        {"action_type": "send_manager_email", "params": {"subject": "직접 호출"}},
        call_id="direct-001",
    )
    assert result["sent"] is True
    assert result["mock"] is True


@pytest.mark.asyncio
async def test_company_db_action_direct():
    handler = CompanyDBAction()
    result = await handler.execute(
        {"action_type": "create_voc_issue", "params": {"tier": "critical"}},
        call_id="direct-002",
    )
    assert result["created"] is True
    assert result["tier"] == "critical"
    assert "issue_id" in result


@pytest.mark.asyncio
async def test_calendar_action_direct():
    handler = CalendarAction()
    result = await handler.execute(
        {"action_type": "schedule_callback", "params": {"title": "직접 재콜"}},
        call_id="direct-003",
    )
    assert result["scheduled"] is True
    assert result["title"] == "직접 재콜"


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
