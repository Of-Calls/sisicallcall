import functools
import time

from langgraph.graph import END, StateGraph

from app.agents.conversational.state import CallState
from app.agents.conversational.nodes.vad_node.vad_node import vad_node
from app.agents.conversational.nodes.speaker_verify_node.speaker_verify_node import speaker_verify_node
from app.agents.conversational.nodes.stt_node.stt_node import stt_node
from app.agents.conversational.nodes.enrollment_node.enrollment_node import enrollment_node
from app.agents.conversational.nodes.norm_text_node.norm_text_node import norm_text_node
from app.agents.conversational.nodes.cache_node.cache_node import cache_node
from app.agents.conversational.nodes.cache_store_node.cache_store_node import cache_store_node
from app.agents.conversational.nodes.intent_router_llm_node.intent_router_llm_node import intent_router_llm_node
from app.agents.conversational.nodes.faq_branch_node.faq_branch_node import faq_branch_node
from app.agents.conversational.nodes.task_branch_node.task_branch_node import task_branch_node
from app.agents.conversational.nodes.auth_branch_node.auth_branch_node import auth_branch_node
from app.agents.conversational.nodes.clarify_branch_node.clarify_branch_node import clarify_branch_node
from app.agents.conversational.nodes.repeat_branch_node.repeat_branch_node import repeat_branch_node
from app.agents.conversational.nodes.escalation_branch_node.escalation_branch_node import escalation_branch_node
from app.agents.conversational.nodes.reviewer_node.reviewer_node import reviewer_node
from app.agents.conversational.nodes.tts_node.tts_node import tts_node
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _timed(name: str):
    """노드 실행 시간 측정 데코레이터 — 5초 응답 제약 디버깅용.

    각 노드 진입~완료 시간을 [node:name] elapsed=NNNms 형식으로 INFO 로그.
    예외 발생 시에도 finally 로 elapsed 측정해 부분 진행 시간 노출.
    """
    def decorator(node_func):
        @functools.wraps(node_func)
        async def wrapped(state):
            t0 = time.monotonic()
            try:
                return await node_func(state)
            finally:
                dt = (time.monotonic() - t0) * 1000
                logger.info(
                    "[node:%s] elapsed=%.0fms call_id=%s",
                    name, dt, state.get("call_id", "?"),
                )
        return wrapped
    return decorator


# ── 조건부 엣지 함수 ──────────────────────────────────────────

def route_after_vad(state: CallState) -> str:
    return "pass" if state["is_speech"] else "skip"


def route_after_speaker_verify(state: CallState) -> str:
    return "pass" if state["is_speaker_verified"] else "reject"


def route_after_stt(state: CallState) -> str:
    return "pass" if state.get("raw_transcript") else "skip"


def route_after_cache(state: CallState) -> str:
    return "hit" if state["cache_hit"] else "miss"


def route_to_branch(state: CallState) -> str:
    """IntentRouterLLM 확정 후 primary_intent → 브랜치."""
    return _intent_to_branch(state["primary_intent"])


def route_after_branch(state: CallState) -> str:
    return "review" if _is_high_risk(state) else "skip_review"


def _intent_to_branch(intent: str | None) -> str:
    mapping = {
        "intent_faq": "faq",
        "intent_task": "task",
        "intent_auth": "auth",
        "intent_clarify": "clarify",
        "intent_repeat": "repeat",
        "intent_escalation": "escalation",
    }
    return mapping.get(intent or "", "escalation")


def _is_high_risk(state: CallState) -> bool:
    # TODO(미배정): 담당자 지정 후 구현 — agents.md Reviewer 섹션 + R-09 결과 확정 후
    return False


# ── 그래프 빌더 ───────────────────────────────────────────────

def build_call_graph():
    graph = StateGraph(CallState)

    # 노드 등록 — 모두 _timed 로 감싸 실행 시간 자동 로깅
    graph.add_node("vad",               _timed("vad")(vad_node))
    graph.add_node("speaker_verify",    _timed("speaker_verify")(speaker_verify_node))
    graph.add_node("stt",               _timed("stt")(stt_node))
    graph.add_node("enrollment",        _timed("enrollment")(enrollment_node))
    graph.add_node("norm_text",         _timed("norm_text")(norm_text_node))
    graph.add_node("cache",             _timed("cache")(cache_node))
    graph.add_node("intent_router_llm", _timed("intent_router_llm")(intent_router_llm_node))
    graph.add_node("faq_branch",        _timed("faq_branch")(faq_branch_node))
    graph.add_node("task_branch",       _timed("task_branch")(task_branch_node))
    graph.add_node("auth_branch",       _timed("auth_branch")(auth_branch_node))
    graph.add_node("clarify_branch",    _timed("clarify_branch")(clarify_branch_node))
    graph.add_node("repeat_branch",     _timed("repeat_branch")(repeat_branch_node))
    graph.add_node("escalation_branch", _timed("escalation_branch")(escalation_branch_node))
    graph.add_node("reviewer",          _timed("reviewer")(reviewer_node))
    graph.add_node("cache_store",       _timed("cache_store")(cache_store_node))
    graph.add_node("tts",               _timed("tts")(tts_node))

    # 진입점
    graph.set_entry_point("vad")

    # 전처리 단계
    graph.add_conditional_edges("vad", route_after_vad,
        {"pass": "speaker_verify", "skip": END})
    graph.add_conditional_edges("speaker_verify", route_after_speaker_verify,
        {"pass": "stt", "reject": END})
    graph.add_conditional_edges("stt", route_after_stt,
        {"pass": "enrollment", "skip": END})
    graph.add_edge("enrollment", "norm_text")
    graph.add_edge("norm_text", "cache")

    # Gate 1 분기 — cache miss 는 IntentRouterLLM 으로 직결
    # (이전 KNN Router 단계는 stub 이라 영구 보류 결정 후 2026-04-27 제거 — CLAUDE.md 참조)
    graph.add_conditional_edges("cache", route_after_cache,
        {"hit": "tts", "miss": "intent_router_llm"})

    # IntentRouterLLM → 브랜치 (clarify 포함 5개)
    graph.add_conditional_edges("intent_router_llm", route_to_branch, {
        "faq": "faq_branch",
        "task": "task_branch",
        "auth": "auth_branch",
        "clarify": "clarify_branch",
        "repeat": "repeat_branch",
        "escalation": "escalation_branch",
    })

    # 브랜치 → (조건부 Reviewer) → cache_store → TTS
    # clarify_branch 도 동일하게 reviewer/cache_store 분기 통과 (cache_store 가 clarify path 차단)
    for branch in ("faq_branch", "task_branch", "auth_branch", "clarify_branch", "repeat_branch", "escalation_branch"):
        graph.add_conditional_edges(branch, route_after_branch,
            {"review": "reviewer", "skip_review": "cache_store"})

    graph.add_edge("reviewer", "cache_store")
    graph.add_edge("cache_store", "tts")
    graph.add_edge("tts", END)

    return graph.compile()
