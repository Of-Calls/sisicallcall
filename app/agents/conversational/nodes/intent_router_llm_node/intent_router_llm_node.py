from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from app.agents.conversational.state import CallState
from app.services.llm.base import BaseLLMService
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_llm: BaseLLMService = GPT4OMiniService()

INTENT_ROUTER_TIMEOUT_SEC = 3.0
VALID_INTENTS = {"intent_faq", "intent_task", "intent_auth", "intent_escalation"}
INTENT_ROUTER_SYSTEM_PROMPT = """You are an intent routing classifier for a customer support call.
Classify the utterance into exactly one of these coarse intents and return JSON only:
- intent_faq
- intent_task
- intent_auth
- intent_escalation

Output format:
{"primary_intent": "intent_xxx", "secondary_intents": [], "routing_reason": "short reason"}"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_user_message(
    normalized_text: str,
    knn_intent: str | None,
    session_view: dict[str, Any],
    knn_top_k: list[dict[str, Any]] | None = None,
) -> str:
    turn_count = session_view.get("turn_count", 0) if session_view else 0
    knn_hint = knn_intent or "none"
    if knn_top_k:
        top_k_hint = json.dumps(knn_top_k, ensure_ascii=False)
    else:
        top_k_hint = "[]"
    return (
        f"User utterance: {normalized_text}\n"
        f"KNN leaf intent hint: {knn_hint}\n"
        f"KNN top-k candidates: {top_k_hint}\n"
        f"Current turn count: {turn_count}"
    )


def _parse_intent_response(raw: str) -> dict[str, Any] | None:
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    intent = parsed.get("primary_intent")
    if intent not in VALID_INTENTS:
        return None

    return {
        "primary_intent": intent,
        "secondary_intents": parsed.get("secondary_intents") or [],
        "routing_reason": parsed.get("routing_reason") or "intent_router_llm",
    }


def _fallback(
    knn_intent: str | None,
    reason: str,
    primary_intent: str | None = None,
    knn_top_k: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fallback_intent = _coerce_primary_intent(primary_intent, knn_intent, knn_top_k)
    if fallback_intent:
        return {
            "primary_intent": fallback_intent,
            "secondary_intents": [],
            "routing_reason": reason,
        }
    return {
        "primary_intent": "intent_escalation",
        "secondary_intents": [],
        "routing_reason": f"{reason}_no_knn",
    }


def _coerce_primary_intent(
    primary_intent: str | None,
    knn_intent: str | None,
    knn_top_k: list[dict[str, Any]] | None,
) -> str | None:
    if primary_intent in VALID_INTENTS:
        return primary_intent

    for candidate in knn_top_k or []:
        branch = candidate.get("branch_intent") or candidate.get("branch")
        if branch in VALID_INTENTS:
            return str(branch)

    if knn_intent in VALID_INTENTS:
        return knn_intent

    return _branch_from_leaf(knn_intent)


def _branch_from_leaf(intent_label: str | None) -> str | None:
    normalized = (intent_label or "").strip()
    for branch_intent in VALID_INTENTS:
        if normalized.startswith(f"{branch_intent}_"):
            return branch_intent
    return None


async def intent_router_llm_node(state: CallState) -> dict:
    knn_intent = state.get("knn_intent")
    primary_intent = state.get("primary_intent")
    knn_top_k = state.get("knn_top_k") or []
    user_message = _build_user_message(
        state["normalized_text"],
        knn_intent,
        state.get("session_view") or {},
        knn_top_k,
    )

    try:
        raw = await asyncio.wait_for(
            _llm.generate(
                system_prompt=INTENT_ROUTER_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.1,
                max_tokens=200,
            ),
            timeout=INTENT_ROUTER_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning("intent_router timeout call_id=%s", state["call_id"])
        return _fallback(knn_intent, "knn_fallback_timeout", primary_intent, knn_top_k)
    except Exception as exc:
        logger.error("intent_router error call_id=%s: %s", state["call_id"], exc)
        return _fallback(knn_intent, "knn_fallback_error", primary_intent, knn_top_k)

    parsed = _parse_intent_response(raw)
    if parsed is None:
        logger.warning(
            "intent_router parse failed call_id=%s raw=%r",
            state["call_id"],
            raw[:200],
        )
        return _fallback(
            knn_intent,
            "knn_fallback_parse_error",
            primary_intent,
            knn_top_k,
        )

    return parsed
