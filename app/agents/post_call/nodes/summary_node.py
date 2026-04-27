from __future__ import annotations

import json

from app.agents.post_call.llm_caller import PostCallLLMCaller, make_summary_caller
from app.agents.post_call.prompts import SUMMARY_SYSTEM, SUMMARY_USER
from app.agents.post_call.schemas import CustomerEmotion, ResolutionStatus, SummaryResult
from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# lazy singleton — 테스트에서 monkeypatch.setattr 으로 교체
_caller: PostCallLLMCaller | None = None

# 녹취 없을 때 반환하는 안전한 fallback 요약 구조
_EMPTY_TRANSCRIPT_SUMMARY: dict = {
    "summary_short": "통화 내용 없음",
    "summary_detailed": "녹취 데이터가 없어 요약을 생성할 수 없습니다.",
    "customer_intent": "알 수 없음",
    "customer_emotion": "neutral",
    "resolution_status": "resolved",
    "keywords": [],
    "handoff_notes": None,
}


def _get_caller() -> PostCallLLMCaller:
    global _caller
    if _caller is None:
        _caller = make_summary_caller()
    return _caller


def _format_transcripts(transcripts: list[dict]) -> str:
    if not transcripts:
        return "(녹취 없음)"
    lines = []
    for t in transcripts:
        role = t.get("role", "unknown")
        text = t.get("text", "")
        lines.append(f"[{role}] {text}")
    return "\n".join(lines)


def _validate(raw: dict) -> dict:
    """Pydantic 으로 스키마 검증 후 직렬화 가능한 dict 반환."""
    # 허용 enum 값 외 값이 오면 기본값으로 fallback
    emotion = raw.get("customer_emotion", "neutral")
    if emotion not in CustomerEmotion._value2member_map_:
        logger.warning("summary: unknown customer_emotion=%r — neutral 로 대체", emotion)
        raw["customer_emotion"] = "neutral"

    status = raw.get("resolution_status", "resolved")
    if status not in ResolutionStatus._value2member_map_:
        logger.warning("summary: unknown resolution_status=%r — resolved 로 대체", status)
        raw["resolution_status"] = "resolved"

    result = SummaryResult(**raw)
    return result.model_dump()


async def summary_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    transcripts: list = state.get("transcripts") or []  # type: ignore[call-overload]

    # 녹취 없음 — LLM 호출 없이 fallback 요약 반환
    if not transcripts:
        logger.warning("summary: 녹취 없음 call_id=%s — fallback 요약 사용", call_id)
        errors = list(state.get("errors", []))  # type: ignore[call-overload]
        errors.append({
            "node": "summary",
            "warning": "empty_transcript",
            "error": "transcripts 없음 — fallback 요약 사용",
        })
        return {"summary": dict(_EMPTY_TRANSCRIPT_SUMMARY), "errors": errors, "partial_success": True}

    try:
        transcripts_text = _format_transcripts(transcripts)
        user_msg = SUMMARY_USER.format(transcripts=transcripts_text)

        raw = await _get_caller().call_json(
            system_prompt=SUMMARY_SYSTEM,
            user_message=user_msg,
            max_tokens=800,
        )
        summary = _validate(raw)
        logger.info(
            "summary 완료 call_id=%s emotion=%s status=%s",
            call_id,
            summary.get("customer_emotion"),
            summary.get("resolution_status"),
        )
        return {"summary": summary}

    except Exception as exc:
        logger.error("summary 실패 call_id=%s err=%s", call_id, exc)
        errors = list(state.get("errors", []))  # type: ignore[call-overload]
        errors.append({"node": "summary", "error": str(exc)})
        return {"summary": None, "errors": errors, "partial_success": True}
