import pytest
from unittest.mock import AsyncMock, patch
from app.agents.post_call.agent import PostCallAgent


@pytest.fixture
def agent():
    return PostCallAgent()


@pytest.mark.asyncio
async def test_run_call_ended_full_pipeline(agent):
    result = await agent.run("call-001", trigger="call_ended", tenant_id="test")

    assert result["call_id"] == "call-001"
    assert result["trigger"] == "call_ended"
    assert result["summary"] is not None
    assert result["voc_analysis"] is not None
    assert result["priority_result"] is not None
    assert result["action_plan"] is not None
    assert isinstance(result["executed_actions"], list)
    assert result["dashboard_payload"] is not None


@pytest.mark.asyncio
async def test_run_escalation_immediate_skips_mcp(agent):
    result = await agent.run("call-002", trigger="escalation_immediate", tenant_id="test")

    assert result["call_id"] == "call-002"
    assert result["trigger"] == "escalation_immediate"
    assert result["summary"] is not None
    # MCP 금지 노드는 실행되지 않음
    assert result["voc_analysis"] is None
    assert result["priority_result"] is None
    assert result["action_plan"] is None
    assert result["executed_actions"] == []
    assert result["dashboard_payload"] is not None


@pytest.mark.asyncio
async def test_run_manual_full_pipeline(agent):
    result = await agent.run("call-003", trigger="manual", tenant_id="test")

    assert result["trigger"] == "manual"
    assert result["summary"] is not None
    assert result["voc_analysis"] is not None
    assert result["priority_result"] is not None


@pytest.mark.asyncio
async def test_invalid_trigger_raises(agent):
    with pytest.raises(ValueError, match="Unknown trigger"):
        await agent.run("call-004", trigger="bad_trigger")


@pytest.mark.asyncio
async def test_errors_list_is_empty_on_success(agent):
    result = await agent.run("call-005", trigger="call_ended")
    assert isinstance(result["errors"], list)
    assert len(result["errors"]) == 0


@pytest.mark.asyncio
async def test_dashboard_payload_has_required_keys(agent):
    result = await agent.run("call-006", trigger="call_ended", tenant_id="demo")
    payload = result["dashboard_payload"]
    assert payload is not None
    for key in ("call_id", "tenant_id", "trigger", "summary", "voc_analysis",
                "priority_result", "action_plan", "executed_actions", "errors", "partial_success"):
        assert key in payload, f"dashboard_payload missing key: {key}"


@pytest.mark.asyncio
async def test_partial_success_false_on_clean_run(agent):
    result = await agent.run("call-007", trigger="call_ended")
    assert result["partial_success"] is False


@pytest.mark.asyncio
async def test_action_plan_contains_actions(agent):
    result = await agent.run("call-008", trigger="call_ended")
    plan = result["action_plan"]
    assert plan is not None
    assert "actions" in plan
    assert len(plan["actions"]) >= 1


@pytest.mark.asyncio
async def test_executed_actions_match_plan(agent):
    result = await agent.run("call-009", trigger="call_ended")
    plan_count = len(result["action_plan"]["actions"])
    exec_count = len(result["executed_actions"])
    assert exec_count == plan_count
