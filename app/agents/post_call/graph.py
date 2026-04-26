from __future__ import annotations
from langgraph.graph import StateGraph, END
from app.agents.post_call.state import PostCallAgentState
from app.agents.post_call.nodes.load_context_node import load_context_node
from app.agents.post_call.nodes.summary_node import summary_node
from app.agents.post_call.nodes.voc_analysis_node import voc_analysis_node
from app.agents.post_call.nodes.priority_node import priority_node
from app.agents.post_call.nodes.action_planner_node import action_planner_node
from app.agents.post_call.nodes.action_router_node import action_router_node
from app.agents.post_call.nodes.save_result_node import save_result_node


def _route_after_summary(state: PostCallAgentState) -> str:
    # escalation_immediate: summary만 실행하고 MCP 금지
    if state["trigger"] == "escalation_immediate":
        return "save_result"
    return "voc_analysis_step"


def build_post_call_graph():
    g: StateGraph = StateGraph(PostCallAgentState)

    # 노드명은 PostCallAgentState 키 (summary, voc_analysis) 와 동명이면
    # LangGraph add_node 가 raise — 그래서 _step 접미어 사용.
    g.add_node("load_context", load_context_node)
    g.add_node("summary_step", summary_node)
    g.add_node("voc_analysis_step", voc_analysis_node)
    g.add_node("priority", priority_node)
    g.add_node("action_planner", action_planner_node)
    g.add_node("action_router", action_router_node)
    g.add_node("save_result", save_result_node)

    g.set_entry_point("load_context")
    g.add_edge("load_context", "summary_step")
    g.add_conditional_edges(
        "summary_step",
        _route_after_summary,
        {"voc_analysis_step": "voc_analysis_step", "save_result": "save_result"},
    )
    g.add_edge("voc_analysis_step", "priority")
    g.add_edge("priority", "action_planner")
    g.add_edge("action_planner", "action_router")
    g.add_edge("action_router", "save_result")
    g.add_edge("save_result", END)

    return g.compile()
