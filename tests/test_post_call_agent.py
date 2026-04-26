"""
PostCallAgent 통합 테스트.

LLM 호출(summary / voc_analysis / priority 노드)은 모두 mock 으로 대체한다.
각 노드 모듈의 모듈-레벨 _caller 를 monkeypatch 로 교체하는 방식을 사용한다.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.post_call.agent import PostCallAgent


# ── Mock 응답 ─────────────────────────────────────────────────────────────────

MOCK_SUMMARY = {
    "summary_short": "요금 문의 상담",
    "summary_detailed": "고객이 요금제 변경을 원했고 상담원이 안내 후 해결됨",
    "customer_intent": "요금제 변경",
    "customer_emotion": "neutral",
    "resolution_status": "resolved",
    "keywords": ["요금", "변경"],
    "handoff_notes": None,
}

MOCK_VOC = {
    "sentiment_result": {
        "sentiment": "negative",
        "intensity": 0.6,
        "reason": "고객이 불만을 표시했으나 해결됨",
    },
    "intent_result": {
        "primary_category": "요금 문의",
        "sub_categories": ["요금제 변경"],
        "is_repeat_topic": False,
        "faq_candidate": True,
    },
    "priority_result": {
        "priority": "high",
        "action_required": True,
        "suggested_action": "VOC 등록 후 팀장 보고",
        "reason": "고객 불만 감지",
    },
}

MOCK_PRIORITY = {
    "priority": "high",
    "tier": "high",
    "action_required": True,
    "suggested_action": "팀장 에스컬레이션",
    "reason": "고객 부정 감정 및 반복 문의",
}


# ── Fixture ───────────────────────────────────────────────────────────────────

def _make_mock_caller(return_map: dict[str, dict]) -> MagicMock:
    """system_prompt 키워드로 응답을 분기하는 mock caller."""
    caller = MagicMock()

    async def _call_json(system_prompt: str, user_message: str, max_tokens: int = 1024) -> dict:
        if "summary_short" in system_prompt:
            return dict(return_map["summary"])
        if "sentiment_result" in system_prompt:
            return dict(return_map["voc"])
        if "tier" in system_prompt and "action_required" in system_prompt:
            return dict(return_map["priority"])
        return {}

    caller.call_json = AsyncMock(side_effect=_call_json)
    return caller


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    """모든 테스트에서 LLM 호출을 mock 으로 대체."""
    mock = _make_mock_caller({
        "summary": MOCK_SUMMARY,
        "voc": MOCK_VOC,
        "priority": MOCK_PRIORITY,
    })
    import app.agents.post_call.nodes.summary_node as sm
    import app.agents.post_call.nodes.voc_analysis_node as vm
    import app.agents.post_call.nodes.priority_node as pm

    monkeypatch.setattr(sm, "_caller", mock)
    monkeypatch.setattr(vm, "_caller", mock)
    monkeypatch.setattr(pm, "_caller", mock)
    return mock


@pytest.fixture
def agent():
    return PostCallAgent()


# ── 기본 플로우 테스트 ─────────────────────────────────────────────────────────

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
    # escalation_immediate: voc/priority/action 노드 실행 안 됨
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


# ── Summary 스키마 검증 ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summary_schema(agent):
    result = await agent.run("call-010", trigger="call_ended")
    s = result["summary"]
    assert s is not None
    for key in ("summary_short", "summary_detailed", "customer_intent",
                "customer_emotion", "resolution_status", "keywords"):
        assert key in s, f"summary missing key: {key}"
    assert s["customer_emotion"] in ("positive", "neutral", "negative", "angry")
    assert s["resolution_status"] in ("resolved", "escalated", "abandoned")
    assert isinstance(s["keywords"], list)


# ── VOC 스키마 검증 ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voc_schema(agent):
    result = await agent.run("call-011", trigger="call_ended")
    voc = result["voc_analysis"]
    assert voc is not None
    assert "sentiment_result" in voc
    assert "intent_result" in voc
    assert "priority_result" in voc

    sr = voc["sentiment_result"]
    assert sr["sentiment"] in ("positive", "neutral", "negative", "angry")
    assert isinstance(sr["intensity"], (int, float))
    assert 0.0 <= sr["intensity"] <= 1.0

    pr = voc["priority_result"]
    assert pr["priority"] in ("low", "medium", "high", "critical")
    assert isinstance(pr["action_required"], bool)


# ── Priority 스키마 검증 ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_priority_schema(agent):
    result = await agent.run("call-012", trigger="call_ended")
    p = result["priority_result"]
    assert p is not None
    assert p["priority"] in ("low", "medium", "high", "critical")
    # action_planner_node 하위 호환
    assert "tier" in p
    assert p["tier"] == p["priority"]
    assert isinstance(p["action_required"], bool)


# ── action_planner tier 하위 호환 ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_action_planner_uses_tier(agent):
    """priority_result['tier'] == 'high' 이면 action_plan 에 send_manager_email 포함."""
    result = await agent.run("call-013", trigger="call_ended")
    assert result["action_plan"] is not None
    action_types = [a["action_type"] for a in result["action_plan"]["actions"]]
    # MOCK_PRIORITY tier == "high" → action_planner 가 send_manager_email 을 추가해야 함
    assert "send_manager_email" in action_types


# ── 오류 경로 테스트 ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summary_llm_failure_partial_success(agent, monkeypatch):
    """summary 노드 LLM 실패 시 partial_success=True, errors 기록."""
    import app.agents.post_call.nodes.summary_node as sm

    failing = MagicMock()
    failing.call_json = AsyncMock(side_effect=ValueError("LLM 오류 시뮬레이션"))
    monkeypatch.setattr(sm, "_caller", failing)

    result = await agent.run("call-020", trigger="call_ended")
    # 에이전트 자체는 크래시하지 않아야 함
    assert result is not None
    assert result["summary"] is None
    assert any("summary" in e.get("node", "") for e in result["errors"])


@pytest.mark.asyncio
async def test_voc_llm_failure_partial_success(agent, monkeypatch):
    """voc_analysis 노드 LLM 실패 시 partial_success=True."""
    import app.agents.post_call.nodes.voc_analysis_node as vm

    failing = MagicMock()
    failing.call_json = AsyncMock(side_effect=RuntimeError("VOC LLM 오류"))
    monkeypatch.setattr(vm, "_caller", failing)

    result = await agent.run("call-021", trigger="call_ended")
    assert result["voc_analysis"] is None
    assert result["partial_success"] is True


# ── 기타 ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_errors_list_empty_on_clean_run(agent):
    result = await agent.run("call-030", trigger="call_ended")
    assert isinstance(result["errors"], list)
    assert len(result["errors"]) == 0


@pytest.mark.asyncio
async def test_dashboard_payload_has_required_keys(agent):
    result = await agent.run("call-031", trigger="call_ended", tenant_id="demo")
    payload = result["dashboard_payload"]
    assert payload is not None
    for key in ("call_id", "tenant_id", "trigger", "summary", "voc_analysis",
                "priority_result", "action_plan", "executed_actions", "errors", "partial_success"):
        assert key in payload, f"dashboard_payload missing key: {key}"


@pytest.mark.asyncio
async def test_llm_called_once_per_node(agent, mock_llm):
    """call_ended 플로우에서 LLM call_json 이 정확히 3회 호출된다 (summary/voc/priority)."""
    await agent.run("call-040", trigger="call_ended")
    assert mock_llm.call_json.call_count == 3


@pytest.mark.asyncio
async def test_escalation_llm_called_once(agent, mock_llm):
    """escalation_immediate 플로우에서 LLM call_json 이 정확히 1회 호출된다 (summary만)."""
    await agent.run("call-041", trigger="escalation_immediate")
    assert mock_llm.call_json.call_count == 1
