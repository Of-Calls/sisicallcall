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
    return "voc_analysis"


def build_post_call_graph():
    g: StateGraph = StateGraph(PostCallAgentState)

    g.add_node("load_context", load_context_node)
    g.add_node("summary", summary_node)
    g.add_node("voc_analysis", voc_analysis_node)
    g.add_node("priority", priority_node)
    g.add_node("action_planner", action_planner_node)
    g.add_node("action_router", action_router_node)
    g.add_node("save_result", save_result_node)

    g.set_entry_point("load_context")
    g.add_edge("load_context", "summary")
    g.add_conditional_edges(
        "summary",
        _route_after_summary,
        {"voc_analysis": "voc_analysis", "save_result": "save_result"},
    )
    g.add_edge("voc_analysis", "priority")
    g.add_edge("priority", "action_planner")
    g.add_edge("action_planner", "action_router")
    g.add_edge("action_router", "save_result")
    g.add_edge("save_result", END)

    return g.compile()
