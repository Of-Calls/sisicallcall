"""
PostCallAgent 통합 테스트 + Action Planner 단위 테스트.

LLM 호출(summary / voc_analysis / priority 노드)은 모두 mock 으로 대체한다.
Action Planner 는 rule-based 이므로 state 를 직접 구성하여 단위 테스트한다.
"""
from __future__ import annotations

import sys
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.post_call.agent import PostCallAgent
from app.agents.post_call.nodes.action_planner_node import action_planner_node


# ── 공통 Mock 응답 ────────────────────────────────────────────────────────────

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
    """system_prompt 키워드로 응답을 분기하는 mock LLM caller."""
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


# ── Action Planner 단위 테스트용 state 빌더 ──────────────────────────────────

def _make_planner_state(
    call_id: str = "t-planner",
    tenant_id: str = "test",
    summary: dict | None = None,
    voc: dict | None = None,
    priority: dict | None = None,
) -> dict:
    """action_planner_node 직접 호출용 최소 state."""
    return {
        "call_id": call_id,
        "tenant_id": tenant_id,
        "trigger": "call_ended",
        "call_metadata": {},
        "transcripts": [],
        "branch_stats": {},
        "summary": summary or {},
        "voc_analysis": voc or {},
        "priority_result": priority or {},
        "action_plan": None,
        "executed_actions": [],
        "dashboard_payload": None,
        "errors": [],
        "partial_success": False,
    }


# ── 헬퍼: action_types 추출 ───────────────────────────────────────────────────

def _action_types(plan: dict) -> list[str]:
    return [a["action_type"] for a in plan.get("actions", [])]


# ── 기본 통합 플로우 테스트 ───────────────────────────────────────────────────

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


# ── 스키마 검증 테스트 ────────────────────────────────────────────────────────

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


@pytest.mark.asyncio
async def test_priority_schema(agent):
    result = await agent.run("call-012", trigger="call_ended")
    p = result["priority_result"]
    assert p is not None
    assert p["priority"] in ("low", "medium", "high", "critical")
    assert "tier" in p
    assert p["tier"] == p["priority"]
    assert isinstance(p["action_required"], bool)


# ── 기존 Action Planner 통합 테스트 (MOCK 기반) ────────────────────────────────

@pytest.mark.asyncio
async def test_action_planner_high_priority_with_faq(agent):
    """MOCK: priority=high + action_required=True + faq_candidate=True
    → create_voc_issue(기본 VOC 등록) + mark_faq_candidate 포함."""
    result = await agent.run("call-013", trigger="call_ended")
    assert result["action_plan"] is not None
    types = _action_types(result["action_plan"])
    assert "create_voc_issue" in types
    assert "mark_faq_candidate" in types


@pytest.mark.asyncio
async def test_action_plan_has_action_required_field(agent):
    """action_plan 에 action_required 필드가 존재해야 한다."""
    result = await agent.run("call-014", trigger="call_ended")
    plan = result["action_plan"]
    assert "action_required" in plan
    assert isinstance(plan["action_required"], bool)


@pytest.mark.asyncio
async def test_action_plan_actions_have_priority_field(agent):
    """각 action item 에 priority 필드가 존재해야 한다."""
    result = await agent.run("call-015", trigger="call_ended")
    for action in result["action_plan"]["actions"]:
        assert "priority" in action
        assert action["priority"] in ("low", "medium", "high", "critical")


# ── 오류 경로 테스트 ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summary_llm_failure_partial_success(agent, monkeypatch):
    import app.agents.post_call.nodes.summary_node as sm
    failing = MagicMock()
    failing.call_json = AsyncMock(side_effect=ValueError("LLM 오류 시뮬레이션"))
    monkeypatch.setattr(sm, "_caller", failing)

    result = await agent.run("call-020", trigger="call_ended")
    assert result is not None
    assert result["summary"] is None
    assert result["partial_success"] is True
    assert any("summary" in e.get("node", "") for e in result["errors"])


@pytest.mark.asyncio
async def test_voc_llm_failure_partial_success(agent, monkeypatch):
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
    await agent.run("call-040", trigger="call_ended")
    assert mock_llm.call_json.call_count == 3


@pytest.mark.asyncio
async def test_escalation_llm_called_once(agent, mock_llm):
    await agent.run("call-041", trigger="escalation_immediate")
    assert mock_llm.call_json.call_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Action Planner 단위 테스트 (LLM 무관 — action_planner_node 직접 호출)
# ═══════════════════════════════════════════════════════════════════════════════

# ── 케이스 1: low + resolved + neutral + faq=false → actions=[] ───────────────

@pytest.mark.asyncio
async def test_planner_no_action_when_all_low():
    """action_required=False + faq_candidate=False 이면 actions=[] (Rule 1 early exit)."""
    state = _make_planner_state(
        call_id="p-001",
        summary={
            "customer_emotion": "neutral",
            "resolution_status": "resolved",
            "handoff_notes": None,
            "summary_short": "일반 문의 완료",
        },
        voc={
            "sentiment_result": {"sentiment": "neutral", "intensity": 0.1, "reason": ""},
            "intent_result": {
                "primary_category": "일반 문의",
                "sub_categories": [],
                "is_repeat_topic": False,
                "faq_candidate": False,
            },
            "priority_result": {
                "priority": "low",
                "action_required": False,
                "suggested_action": None,
                "reason": "",
            },
        },
        priority={
            "priority": "low",
            "tier": "low",
            "action_required": False,
            "suggested_action": None,
            "reason": "단순 문의 해결",
        },
    )
    result = await action_planner_node(state)
    plan = result["action_plan"]
    assert plan is not None
    assert plan["actions"] == []
    assert plan["action_required"] is False


# ── 케이스 2: high + negative + unresolved → create_voc_issue ─────────────────

@pytest.mark.asyncio
async def test_planner_high_negative_unresolved_creates_voc():
    """priority=high + action_required=True + negative + abandoned → create_voc_issue 포함."""
    state = _make_planner_state(
        call_id="p-002",
        summary={
            "customer_emotion": "negative",
            "resolution_status": "abandoned",
            "handoff_notes": None,
            "summary_short": "미해결 종료",
        },
        voc={
            "sentiment_result": {"sentiment": "negative", "intensity": 0.6, "reason": ""},
            "intent_result": {
                "primary_category": "요금 불만",
                "sub_categories": [],
                "is_repeat_topic": False,
                "faq_candidate": False,
            },
            "priority_result": {
                "priority": "high",
                "action_required": True,
                "suggested_action": None,
                "reason": "",
            },
        },
        priority={
            "priority": "high",
            "tier": "high",
            "action_required": True,
            "suggested_action": None,
            "reason": "미해결 고객 불만",
        },
    )
    result = await action_planner_node(state)
    types = _action_types(result["action_plan"])
    assert "create_voc_issue" in types


# ── 케이스 3: angry + escalated → 3개 액션 ────────────────────────────────────

@pytest.mark.asyncio
async def test_planner_angry_escalated_three_actions():
    """angry + escalated 조합은 create_voc_issue + send_manager_email + add_priority_queue 를 생성 (Rule 3)."""
    state = _make_planner_state(
        call_id="p-003",
        summary={
            "customer_emotion": "angry",
            "resolution_status": "escalated",
            "handoff_notes": None,
            "summary_short": "강한 불만 에스컬레이션",
        },
        voc={
            "sentiment_result": {"sentiment": "angry", "intensity": 0.9, "reason": ""},
            "intent_result": {
                "primary_category": "서비스 장애",
                "sub_categories": [],
                "is_repeat_topic": False,
                "faq_candidate": False,
            },
            "priority_result": {
                "priority": "high",
                "action_required": True,
                "suggested_action": None,
                "reason": "",
            },
        },
        priority={
            "priority": "high",
            "tier": "high",
            "action_required": True,
            "suggested_action": None,
            "reason": "강한 불만",
        },
    )
    result = await action_planner_node(state)
    types = _action_types(result["action_plan"])
    assert "create_voc_issue" in types
    assert "send_manager_email" in types
    assert "add_priority_queue" in types


# ── 케이스 4: critical priority → send_manager_email 필수 ─────────────────────

@pytest.mark.asyncio
async def test_planner_critical_must_include_email():
    """priority=critical 이면 send_manager_email 이 반드시 포함되어야 한다 (Rule 5)."""
    state = _make_planner_state(
        call_id="p-004",
        summary={
            "customer_emotion": "neutral",
            "resolution_status": "escalated",
            "handoff_notes": None,
            "summary_short": "법적 분쟁 위협",
        },
        voc={
            "sentiment_result": {"sentiment": "angry", "intensity": 0.95, "reason": ""},
            "intent_result": {
                "primary_category": "법적 분쟁",
                "sub_categories": [],
                "is_repeat_topic": False,
                "faq_candidate": False,
            },
            "priority_result": {
                "priority": "critical",
                "action_required": True,
                "suggested_action": None,
                "reason": "",
            },
        },
        priority={
            "priority": "critical",
            "tier": "critical",
            "action_required": True,
            "suggested_action": None,
            "reason": "즉시 대응 필요",
        },
    )
    result = await action_planner_node(state)
    types = _action_types(result["action_plan"])
    assert "send_manager_email" in types


# ── 케이스 5: faq_candidate=true → mark_faq_candidate ────────────────────────

@pytest.mark.asyncio
async def test_planner_faq_candidate_includes_mark():
    """faq_candidate=True 이면 action_required=False 여도 mark_faq_candidate 포함 (Rule 2 / 7)."""
    state = _make_planner_state(
        call_id="p-005",
        summary={
            "customer_emotion": "neutral",
            "resolution_status": "resolved",
            "handoff_notes": None,
            "summary_short": "FAQ 후보 상담",
        },
        voc={
            "sentiment_result": {"sentiment": "neutral", "intensity": 0.1, "reason": ""},
            "intent_result": {
                "primary_category": "요금제 변경 방법",
                "sub_categories": [],
                "is_repeat_topic": False,
                "faq_candidate": True,
            },
            "priority_result": {
                "priority": "low",
                "action_required": False,
                "suggested_action": None,
                "reason": "",
            },
        },
        priority={
            "priority": "low",
            "tier": "low",
            "action_required": False,
            "suggested_action": None,
            "reason": "일반 처리",
        },
    )
    result = await action_planner_node(state)
    types = _action_types(result["action_plan"])
    assert "mark_faq_candidate" in types


# ── 케이스 6: 콜백 키워드 포함 → schedule_callback ───────────────────────────

@pytest.mark.asyncio
async def test_planner_callback_keyword_in_handoff_notes():
    """handoff_notes 에 '콜백' 키워드가 있으면 schedule_callback 포함 (Rule 6)."""
    state = _make_planner_state(
        call_id="p-006",
        summary={
            "customer_emotion": "neutral",
            "resolution_status": "resolved",
            "handoff_notes": "고객에게 콜백 예약 필요",
            "summary_short": "콜백 요청",
        },
        voc={
            "sentiment_result": {"sentiment": "neutral", "intensity": 0.2, "reason": ""},
            "intent_result": {
                "primary_category": "콜백 요청",
                "sub_categories": [],
                "is_repeat_topic": False,
                "faq_candidate": False,
            },
            "priority_result": {
                "priority": "medium",
                "action_required": True,
                "suggested_action": None,
                "reason": "",
            },
        },
        priority={
            "priority": "medium",
            "tier": "medium",
            "action_required": True,
            "suggested_action": None,
            "reason": "콜백 처리 필요",
        },
    )
    result = await action_planner_node(state)
    types = _action_types(result["action_plan"])
    assert "schedule_callback" in types


@pytest.mark.asyncio
async def test_planner_callback_keyword_in_suggested_action():
    """suggested_action 에 '다시 전화' 키워드가 있으면 schedule_callback 포함 (Rule 6)."""
    state = _make_planner_state(
        call_id="p-006b",
        summary={
            "customer_emotion": "neutral",
            "resolution_status": "resolved",
            "handoff_notes": None,
            "summary_short": "재연락 필요",
        },
        voc={
            "sentiment_result": {"sentiment": "neutral", "intensity": 0.3, "reason": ""},
            "intent_result": {
                "primary_category": "재연락 요청",
                "sub_categories": [],
                "is_repeat_topic": False,
                "faq_candidate": False,
            },
            "priority_result": {
                "priority": "medium",
                "action_required": True,
                "suggested_action": "고객에게 다시 전화 필요",
                "reason": "",
            },
        },
        priority={
            "priority": "medium",
            "tier": "medium",
            "action_required": True,
            "suggested_action": "고객에게 다시 전화 필요",
            "reason": "재연락 처리",
        },
    )
    result = await action_planner_node(state)
    types = _action_types(result["action_plan"])
    assert "schedule_callback" in types


# ── 케이스 7: 중복 조건 겹침 → action_type 중복 제거 ────────────────────────

@pytest.mark.asyncio
async def test_planner_no_duplicate_action_types():
    """Rule 3 + Rule 4 + Rule 5 가 동시에 트리거되어도 action_type 은 중복되지 않는다 (Rule 8)."""
    state = _make_planner_state(
        call_id="p-007",
        summary={
            "customer_emotion": "angry",
            "resolution_status": "escalated",
            "handoff_notes": None,
            "summary_short": "복합 트리거",
        },
        voc={
            "sentiment_result": {"sentiment": "angry", "intensity": 0.95, "reason": ""},
            "intent_result": {
                "primary_category": "반복 불만",
                "sub_categories": [],
                "is_repeat_topic": True,   # Rule 4 도 트리거
                "faq_candidate": False,
            },
            "priority_result": {
                "priority": "critical",
                "action_required": True,
                "suggested_action": None,
                "reason": "",
            },
        },
        priority={
            "priority": "critical",       # Rule 5 도 트리거
            "tier": "critical",
            "action_required": True,
            "suggested_action": None,
            "reason": "즉시 대응",
        },
    )
    result = await action_planner_node(state)
    types = _action_types(result["action_plan"])

    # 중복 없음
    assert len(types) == len(set(types)), f"중복 action_type 발견: {types}"
    # Rule 3 에 의한 핵심 액션 포함 확인
    assert "create_voc_issue" in types
    assert "send_manager_email" in types
    assert "add_priority_queue" in types


# ── 추가: action_plan 구조 검증 ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_planner_action_params_contain_call_id():
    """모든 action 의 params 에 call_id 가 포함되어야 한다."""
    state = _make_planner_state(
        call_id="p-010",
        summary={
            "customer_emotion": "angry",
            "resolution_status": "escalated",
            "handoff_notes": None,
            "summary_short": "에스컬레이션",
        },
        voc={
            "sentiment_result": {"sentiment": "angry", "intensity": 0.8, "reason": ""},
            "intent_result": {"primary_category": "불만", "sub_categories": [], "is_repeat_topic": False, "faq_candidate": False},
            "priority_result": {"priority": "high", "action_required": True, "suggested_action": None, "reason": ""},
        },
        priority={"priority": "high", "tier": "high", "action_required": True, "suggested_action": None, "reason": ""},
    )
    result = await action_planner_node(state)
    for action in result["action_plan"]["actions"]:
        assert "call_id" in action["params"], f"{action['action_type']} params 에 call_id 없음"
        assert action["params"]["call_id"] == "p-010"


@pytest.mark.asyncio
async def test_planner_tool_mapping_is_valid():
    """각 action 의 tool 값이 허용된 enum 값이어야 한다."""
    allowed_tools = {"company_db", "gmail", "calendar", "internal_dashboard"}
    state = _make_planner_state(
        call_id="p-011",
        summary={
            "customer_emotion": "angry",
            "resolution_status": "escalated",
            "handoff_notes": "콜백 필요",
            "summary_short": "복합",
        },
        voc={
            "sentiment_result": {"sentiment": "angry", "intensity": 0.9, "reason": ""},
            "intent_result": {"primary_category": "불만", "sub_categories": [], "is_repeat_topic": False, "faq_candidate": True},
            "priority_result": {"priority": "critical", "action_required": True, "suggested_action": None, "reason": ""},
        },
        priority={"priority": "critical", "tier": "critical", "action_required": True, "suggested_action": None, "reason": ""},
    )
    result = await action_planner_node(state)
    for action in result["action_plan"]["actions"]:
        assert action["tool"] in allowed_tools, f"허용되지 않은 tool: {action['tool']}"
        assert action["status"] == "pending"


# ═══════════════════════════════════════════════════════════════════════════════
# 요구사항 2: empty transcript 처리
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_empty_transcript_completes(agent, monkeypatch):
    """transcripts 가 비어 있어도 agent.run() 이 끝까지 완료된다."""
    import app.agents.post_call.nodes.load_context_node as lcn

    mock_repo = MagicMock()
    mock_repo.get_call_context = AsyncMock(return_value={
        "metadata": {"call_id": "empty-001"},
        "transcripts": [],
        "branch_stats": {},
    })
    monkeypatch.setattr(lcn, "_repo", mock_repo)

    result = await agent.run("empty-001", trigger="call_ended")

    assert result is not None
    assert result["partial_success"] is True
    assert any(
        e.get("warning") == "empty_transcript" or "empty" in e.get("error", "").lower()
        for e in result["errors"]
    ), f"empty_transcript 에러 없음: {result['errors']}"


@pytest.mark.asyncio
async def test_empty_transcript_escalation_completes(agent, monkeypatch):
    """escalation_immediate + 빈 녹취도 끝까지 완료된다."""
    import app.agents.post_call.nodes.load_context_node as lcn

    mock_repo = MagicMock()
    mock_repo.get_call_context = AsyncMock(return_value={
        "metadata": {"call_id": "empty-002"},
        "transcripts": [],
        "branch_stats": {},
    })
    monkeypatch.setattr(lcn, "_repo", mock_repo)

    result = await agent.run("empty-002", trigger="escalation_immediate")

    assert result is not None
    assert result["partial_success"] is True
    assert result["voc_analysis"] is None
    assert result["action_plan"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# 요구사항 3: LLM 실패 처리
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_priority_llm_failure_partial_success(agent, monkeypatch):
    """priority_node LLM 실패 시 save_result 까지 도달하고 partial_success=True."""
    import app.agents.post_call.nodes.priority_node as pm
    failing = MagicMock()
    failing.call_json = AsyncMock(side_effect=RuntimeError("Priority LLM 오류"))
    monkeypatch.setattr(pm, "_caller", failing)

    result = await agent.run("call-022", trigger="call_ended")

    assert result is not None
    assert result["priority_result"] is None
    assert result["partial_success"] is True
    assert any("priority" in e.get("node", "") for e in result["errors"])
    assert result["dashboard_payload"] is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 요구사항 5: action_router 안전 처리
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_action_router_none_plan_safe():
    """action_plan=None 이어도 action_router 는 executed_actions=[] 를 반환하고 예외를 던지지 않는다."""
    from app.agents.post_call.nodes.action_router_node import action_router_node

    state = _make_planner_state(call_id="router-001")
    state["action_plan"] = None
    state["errors"] = [{"node": "action_planner", "error": "플래너 실패 시뮬레이션"}]

    result = await action_router_node(state)

    assert isinstance(result, dict)
    assert result.get("executed_actions") == []
    # partial_success 를 강제로 True 로 바꾸지 않는다 (save_result 가 권위 있는 setter)
    assert "partial_success" not in result


@pytest.mark.asyncio
async def test_action_router_empty_actions_safe():
    """action_plan.actions=[] 이어도 action_router 는 정상 동작한다."""
    from app.agents.post_call.nodes.action_router_node import action_router_node

    state = _make_planner_state(call_id="router-002")
    state["action_plan"] = {"action_required": False, "actions": [], "rationale": "no action"}

    result = await action_router_node(state)

    assert result.get("executed_actions") == []
    assert "partial_success" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# 요구사항 4: dashboard_payload.partial_success 일관성
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_dashboard_partial_success_consistent_clean(agent):
    """정상 실행 시 dashboard_payload.partial_success 와 result.partial_success 가 일치한다."""
    result = await agent.run("consist-001", trigger="call_ended")

    assert result["dashboard_payload"] is not None
    assert result["dashboard_payload"]["partial_success"] == result["partial_success"]


@pytest.mark.asyncio
async def test_dashboard_partial_success_consistent_failure(agent, monkeypatch):
    """LLM 실패 시에도 dashboard_payload.partial_success 와 result.partial_success 가 일치한다."""
    import app.agents.post_call.nodes.voc_analysis_node as vm
    failing = MagicMock()
    failing.call_json = AsyncMock(side_effect=RuntimeError("일관성 테스트용 VOC 오류"))
    monkeypatch.setattr(vm, "_caller", failing)

    result = await agent.run("consist-002", trigger="call_ended")

    assert result["dashboard_payload"]["partial_success"] == result["partial_success"]
    assert result["partial_success"] is True


@pytest.mark.asyncio
async def test_dashboard_payload_contains_all_keys(agent):
    """dashboard_payload 에 summary, voc_analysis, priority_result, action_plan,
    executed_actions, errors 가 모두 포함되어야 한다."""
    result = await agent.run("payload-001", trigger="call_ended")
    payload = result["dashboard_payload"]
    assert payload is not None
    for key in ("summary", "voc_analysis", "priority_result", "action_plan",
                "executed_actions", "errors", "partial_success"):
        assert key in payload, f"dashboard_payload 에 {key!r} 없음"


# ═══════════════════════════════════════════════════════════════════════════════
# 요구사항 6: scripts/run_post_call_agent.py Mock 실행
# ═══════════════════════════════════════════════════════════════════════════════

def test_run_post_call_script_mock():
    """scripts/run_post_call_agent.py 가 Mock LLM 으로 정상 종료된다 (returncode=0)."""
    import subprocess
    import os
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "scripts/run_post_call_agent.py", "--trigger", "call_ended"],
        capture_output=True,
        timeout=30,
        cwd=str(project_root),
        env=env,
    )
    assert result.returncode == 0, (
        f"스크립트 실행 실패 (returncode={result.returncode})\n"
        f"stderr: {result.stderr.decode('utf-8', errors='replace')}"
    )


def test_run_post_call_script_escalation_mock():
    """--trigger escalation_immediate 도 정상 종료된다."""
    import subprocess
    import os
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "scripts/run_post_call_agent.py", "--trigger", "escalation_immediate"],
        capture_output=True,
        timeout=30,
        cwd=str(project_root),
        env=env,
    )
    assert result.returncode == 0, (
        f"escalation_immediate 스크립트 실패\n"
        f"stderr: {result.stderr.decode('utf-8', errors='replace')}"
    )
