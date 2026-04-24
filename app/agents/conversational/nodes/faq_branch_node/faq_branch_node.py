from __future__ import annotations

from app.agents.conversational.state import CallState
from app.agents.conversational.utils.stall import FALLBACK_MESSAGE, _run_with_stall
from app.services.llm.base import BaseLLMService
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.rag.base import BaseRAGService
from app.services.rag.chroma import ChromaRAGService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_rag: BaseRAGService | None = None
_llm: BaseLLMService | None = None

FAQ_LLM_TIMEOUT_SEC = 3.0
RAG_TOP_K = 3
FAQ_SYSTEM_PROMPT = """You are an FAQ answer assistant for a customer support call.
Answer only from the provided reference material.
If there is not enough grounded context, respond with:
"Let me check that for you and connect you to the appropriate staff member."
Keep the answer concise."""


def _compose_user_message(normalized_text: str, rag_results: list[str]) -> str:
    if rag_results:
        joined = "\n\n".join(f"[{index + 1}] {chunk}" for index, chunk in enumerate(rag_results))
    else:
        joined = "(no reference material)"
    return f"Reference material:\n{joined}\n\nCustomer question: {normalized_text}"


def _pick_stall_msg(state: CallState) -> str:
    msgs = state.get("stall_messages") or {}
    return msgs.get("faq") or msgs.get("general") or "Please wait a moment while I check."


def _get_rag_service() -> BaseRAGService:
    global _rag
    if _rag is None:
        _rag = ChromaRAGService()
    return _rag


def _get_llm_service() -> BaseLLMService:
    global _llm
    if _llm is None:
        _llm = GPT4OMiniService()
    return _llm


async def faq_branch_node(state: CallState) -> dict:
    call_id = state["call_id"]
    query_embedding = state.get("query_embedding") or []
    leaf_intent_hint = (state.get("knn_intent") or "").strip()
    if leaf_intent_hint:
        logger.debug(
            "faq branch leaf intent hint call_id=%s knn_intent=%s",
            call_id,
            leaf_intent_hint,
        )

    rag_results: list[str] = []
    if query_embedding:
        try:
            rag_results = await _get_rag_service().search(
                query_embedding=query_embedding,
                tenant_id=state["tenant_id"],
                top_k=RAG_TOP_K,
            )
        except Exception as exc:
            logger.error("rag search failed call_id=%s: %s", call_id, exc)

    user_message = _compose_user_message(state["normalized_text"], rag_results)

    response_text, is_timeout = await _run_with_stall(
        coro=_get_llm_service().generate(
            system_prompt=FAQ_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.1,
            max_tokens=300,
        ),
        call_id=call_id,
        stall_msg=_pick_stall_msg(state),
        stall_audio_field="faq",
        delay=state.get("stall_delay_sec", 1.0),
        hardcut_sec=FAQ_LLM_TIMEOUT_SEC,
        rag_results=rag_results,
        fallback_text=FALLBACK_MESSAGE,
    )

    if not response_text:
        response_text = FALLBACK_MESSAGE

    is_fallback = (not rag_results) or (response_text == FALLBACK_MESSAGE)

    return {
        "rag_results": rag_results,
        "response_text": response_text,
        "response_path": "faq",
        "is_timeout": is_timeout,
        "is_fallback": is_fallback,
    }
