import pytest
from app.agents.post_call.actions.executor import ActionExecutor, execute_actions
from app.agents.post_call.actions.gmail_action import GmailAction
from app.agents.post_call.actions.company_db_action import CompanyDBAction
from app.agents.post_call.actions.calendar_action import CalendarAction
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


# ── 새 execute_actions 인터페이스 테스트 (10개) ──────────────────────────────

@pytest.mark.asyncio
async def test_execute_actions_module_level_gmail():
    """모듈 레벨 execute_actions 함수가 gmail 액션을 성공 처리한다."""
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
    """actions=None 이면 빈 리스트를 반환한다."""
    results = await execute_actions(call_id="ea-002", tenant_id="t", actions=None)
    assert results == []


@pytest.mark.asyncio
async def test_execute_actions_empty_list_returns_empty():
    """actions=[] 이면 빈 리스트를 반환한다."""
    results = await execute_actions(call_id="ea-003", tenant_id="t", actions=[])
    assert results == []


@pytest.mark.asyncio
async def test_execute_actions_standard_6_key_format():
    """결과가 표준 6-key 포맷(action_type, tool, status, external_id, error, result)을 갖는다."""
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
    """GmailAction external_id 는 gmail-mock-{call_id} 형식이어야 한다."""
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
    """CompanyDBAction external_id 는 VOC-MOCK-{call_id} 형식이어야 한다."""
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
    """CalendarAction external_id 는 calendar-mock-{call_id} 형식이어야 한다."""
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
    """CompanyDBAction result 에 priority, primary_category, reason, summary_short 가 포함된다."""
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
    """execute_actions 경유 시 알 수 없는 tool 도 failed 여야 한다."""
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
    """한 액션 실패가 나머지 실행을 막지 않는다."""
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
