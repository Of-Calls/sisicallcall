from __future__ import annotations
from langgraph.graph import StateGraph, END
from app.agents.post_call.state import PostCallAgentState
from app.agents.post_call.nodes.load_context_node import load_context_node
from app.agents.post_call.nodes.post_call_analysis_node import post_call_analysis_node
from app.agents.post_call.nodes.review_node import review_node
from app.agents.post_call.nodes.apply_review_corrections_node import apply_review_corrections_node
from app.agents.post_call.nodes.review_control_node import (
    increment_review_retry_node,
    mark_human_review_required_node,
)
from app.agents.post_call.nodes.action_planner_node import action_planner_node
from app.agents.post_call.nodes.action_router_node import action_router_node
from app.agents.post_call.nodes.save_result_node import save_result_node


def _route_after_analysis(state: PostCallAgentState) -> str:
    # escalation_immediate: 통합 분석만 실행하고 MCP action 없이 저장
    if state["trigger"] == "escalation_immediate":
        return "save_result"
    return "review_step"


def _route_after_review(state: PostCallAgentState) -> str:
    verdict = (
        state.get("review_verdict")  # type: ignore[call-overload]
        or (state.get("review_result") or {}).get("verdict")  # type: ignore[call-overload]
        or "fail"
    )
    retry_count = int(state.get("review_retry_count") or 0)  # type: ignore[call-overload]

    if verdict == "pass":
        return "action_planner"
    if verdict == "correctable":
        return "apply_review_corrections_step"
    if verdict == "retry" and retry_count < 1:
        return "increment_review_retry_step"
    return "mark_human_review_required_step"


def build_post_call_graph():
    g: StateGraph = StateGraph(PostCallAgentState)

    # 노드명은 PostCallAgentState 키와 동명이면 LangGraph add_node 가 raise —
    # 그래서 state key 와 겹치는 이름에는 _step 접미어 사용.
    g.add_node("load_context", load_context_node)
    g.add_node("post_call_analysis_step", post_call_analysis_node)
    g.add_node("review_step", review_node)
    g.add_node("apply_review_corrections_step", apply_review_corrections_node)
    g.add_node("increment_review_retry_step", increment_review_retry_node)
    g.add_node("mark_human_review_required_step", mark_human_review_required_node)
    g.add_node("action_planner", action_planner_node)
    g.add_node("action_router", action_router_node)
    g.add_node("save_result", save_result_node)

    g.set_entry_point("load_context")
    g.add_edge("load_context", "post_call_analysis_step")
    g.add_conditional_edges(
        "post_call_analysis_step",
        _route_after_analysis,
        {"review_step": "review_step", "save_result": "save_result"},
    )
    g.add_conditional_edges(
        "review_step",
        _route_after_review,
        {
            "action_planner": "action_planner",
            "apply_review_corrections_step": "apply_review_corrections_step",
            "increment_review_retry_step": "increment_review_retry_step",
            "mark_human_review_required_step": "mark_human_review_required_step",
        },
    )
    g.add_edge("increment_review_retry_step", "post_call_analysis_step")
    g.add_edge("apply_review_corrections_step", "action_planner")
    g.add_edge("action_planner", "action_router")
    g.add_edge("action_router", "save_result")
    g.add_edge("mark_human_review_required_step", "save_result")
    g.add_edge("save_result", END)

    return g.compile()
