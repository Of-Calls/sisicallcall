from langgraph.graph import END, StateGraph

from app.agents.conversational.state import CallState


def route_after_vad(state: CallState) -> str:
    return "pass" if state["is_speech"] else "skip"


def route_after_speaker_verify(state: CallState) -> str:
    return "pass" if state["is_speaker_verified"] else "reject"


def route_after_stt(state: CallState) -> str:
    return "pass" if state["raw_transcript"] else "skip"


def route_after_cache(state: CallState) -> str:
    return "hit" if state["cache_hit"] else "miss"


def route_after_knn(state: CallState) -> str:
    branch = _intent_to_branch(state.get("primary_intent"))
    if branch:
        return branch
    return "intent_router_llm"


def route_to_branch(state: CallState) -> str:
    return _intent_to_branch(state.get("primary_intent")) or "escalation"


def route_after_branch(state: CallState) -> str:
    return "review" if _is_high_risk(state) else "skip_review"


def _intent_to_branch(intent: str | None) -> str | None:
    mapping = {
        "intent_faq": "faq",
        "intent_task": "task",
        "intent_auth": "auth",
        "intent_escalation": "escalation",
    }
    return mapping.get(intent or "")


def _is_high_risk(state: CallState) -> bool:
    return False


def build_call_graph():
    from app.agents.conversational.nodes.auth_branch_node.auth_branch_node import auth_branch_node
    from app.agents.conversational.nodes.cache_node.cache_node import cache_node
    from app.agents.conversational.nodes.cache_store_node.cache_store_node import (
        cache_store_node,
    )
    from app.agents.conversational.nodes.escalation_branch_node.escalation_branch_node import (
        escalation_branch_node,
    )
    from app.agents.conversational.nodes.faq_branch_node.faq_branch_node import faq_branch_node
    from app.agents.conversational.nodes.intent_router_llm_node.intent_router_llm_node import (
        intent_router_llm_node,
    )
    from app.agents.conversational.nodes.knn_router_node.knn_router_node import knn_router_node
    from app.agents.conversational.nodes.norm_text_node.norm_text_node import norm_text_node
    from app.agents.conversational.nodes.reviewer_node.reviewer_node import reviewer_node
    from app.agents.conversational.nodes.speaker_verify_node.speaker_verify_node import (
        speaker_verify_node,
    )
    from app.agents.conversational.nodes.stt_node.stt_node import stt_node
    from app.agents.conversational.nodes.task_branch_node.task_branch_node import (
        task_branch_node,
    )
    from app.agents.conversational.nodes.tts_node.tts_node import tts_node
    from app.agents.conversational.nodes.vad_node.vad_node import vad_node

    graph = StateGraph(CallState)

    graph.add_node("vad", vad_node)
    graph.add_node("speaker_verify", speaker_verify_node)
    graph.add_node("stt", stt_node)
    graph.add_node("norm_text", norm_text_node)
    graph.add_node("cache", cache_node)
    graph.add_node("knn_router", knn_router_node)
    graph.add_node("intent_router_llm", intent_router_llm_node)
    graph.add_node("faq_branch", faq_branch_node)
    graph.add_node("task_branch", task_branch_node)
    graph.add_node("auth_branch", auth_branch_node)
    graph.add_node("escalation_branch", escalation_branch_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("cache_store", cache_store_node)
    graph.add_node("tts", tts_node)

    graph.set_entry_point("vad")

    graph.add_conditional_edges("vad", route_after_vad, {"pass": "speaker_verify", "skip": END})
    graph.add_conditional_edges(
        "speaker_verify",
        route_after_speaker_verify,
        {"pass": "stt", "reject": END},
    )
    graph.add_conditional_edges("stt", route_after_stt, {"pass": "norm_text", "skip": END})
    graph.add_edge("norm_text", "cache")

    graph.add_conditional_edges("cache", route_after_cache, {"hit": "tts", "miss": "knn_router"})

    graph.add_conditional_edges(
        "knn_router",
        route_after_knn,
        {
            "faq": "faq_branch",
            "task": "task_branch",
            "auth": "auth_branch",
            "escalation": "escalation_branch",
            "intent_router_llm": "intent_router_llm",
        },
    )

    graph.add_conditional_edges(
        "intent_router_llm",
        route_to_branch,
        {
            "faq": "faq_branch",
            "task": "task_branch",
            "auth": "auth_branch",
            "escalation": "escalation_branch",
        },
    )

    for branch in ("faq_branch", "task_branch", "auth_branch", "escalation_branch"):
        graph.add_conditional_edges(
            branch,
            route_after_branch,
            {"review": "reviewer", "skip_review": "cache_store"},
        )

    graph.add_edge("reviewer", "cache_store")
    graph.add_edge("cache_store", "tts")
    graph.add_edge("tts", END)

    return graph.compile()
