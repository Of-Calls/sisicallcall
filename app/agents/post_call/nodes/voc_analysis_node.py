from __future__ import annotations

import json

from app.agents.post_call.llm_caller import PostCallLLMCaller, make_voc_caller
from app.agents.post_call.prompts import VOC_SYSTEM, VOC_USER
from app.agents.post_call.schemas import CustomerEmotion, PriorityLevel, VOCResult
from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)

_caller: PostCallLLMCaller | None = None


def _get_caller() -> PostCallLLMCaller:
    global _caller
    if _caller is None:
        _caller = make_voc_caller()
    return _caller


def _format_transcripts(transcripts: list[dict]) -> str:
    if not transcripts:
        return "(녹취 없음)"
    return "\n".join(f"[{t.get('role','?')}] {t.get('text','')}" for t in transcripts)


def _validate(raw: dict) -> dict:
    """Pydantic 검증 + enum fallback."""
    # sentiment
    sr = raw.get("sentiment_result", {})
    if not isinstance(sr, dict):
        sr = {}
    if sr.get("sentiment") not in CustomerEmotion._value2member_map_:
        logger.warning("voc: unknown sentiment=%r — neutral 로 대체", sr.get("sentiment"))
        sr["sentiment"] = "neutral"

    # priority
    pr = raw.get("priority_result", {})
    if not isinstance(pr, dict):
        pr = {}
    if pr.get("priority") not in PriorityLevel._value2member_map_:
        logger.warning("voc: unknown priority=%r — low 로 대체", pr.get("priority"))
        pr["priority"] = "low"

    # intent
    ir = raw.get("intent_result", {})
    if not isinstance(ir, dict):
        ir = {"primary_category": "알 수 없음"}

    raw["sentiment_result"] = sr
    raw["priority_result"] = pr
    raw["intent_result"] = ir

    result = VOCResult(**raw)
    return result.model_dump()


async def voc_analysis_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    try:
        summary = state.get("summary") or {}  # type: ignore[call-overload]
        transcripts_text = _format_transcripts(state.get("transcripts", []))  # type: ignore[call-overload]

        summary_text = summary.get("summary_detailed") or summary.get("summary_short") or "(요약 없음)"
        user_msg = VOC_USER.format(
            summary=summary_text,
            transcripts=transcripts_text,
        )

        raw = await _get_caller().call_json(
            system_prompt=VOC_SYSTEM,
            user_message=user_msg,
            max_tokens=800,
        )
        voc = _validate(raw)
        logger.info(
            "voc_analysis 완료 call_id=%s sentiment=%s priority=%s",
            call_id,
            voc.get("sentiment_result", {}).get("sentiment"),
            voc.get("priority_result", {}).get("priority"),
        )
        return {"voc_analysis": voc}

    except Exception as exc:
        logger.error("voc_analysis 실패 call_id=%s err=%s", call_id, exc)
        errors = list(state.get("errors", []))  # type: ignore[call-overload]
        errors.append({"node": "voc_analysis", "error": str(exc)})
        return {"voc_analysis": None, "errors": errors, "partial_success": True}
