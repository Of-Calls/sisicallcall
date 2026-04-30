"""
Review Gate 노드.

통화 녹취와 post_call_analysis_node 결과를 비교하여
분석이 원문에 근거하는지 검토하고 verdict를 결정한다.

verdict:
  pass        → action_planner 진행
  correctable → apply_review_corrections_node 거쳐 action_planner
  retry       → review_retry_count < 1 이면 재분석
  fail        → human_review_required=True, 외부 action 금지
"""
from __future__ import annotations

import json

from app.agents.post_call.llm_caller import PostCallLLMCaller, make_review_caller
from app.agents.post_call.prompts import REVIEW_SYSTEM, REVIEW_USER
from app.agents.post_call.schemas import ReviewVerdictValues
from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# lazy singleton — 테스트에서 monkeypatch.setattr 으로 교체
_caller: PostCallLLMCaller | None = None


def _get_caller() -> PostCallLLMCaller:
    global _caller
    if _caller is None:
        _caller = make_review_caller()
    return _caller


def _format_transcripts(transcripts: list[dict]) -> str:
    if not transcripts:
        return "(녹취 없음)"
    return "\n".join(f"[{t.get('role', '?')}] {t.get('text', '')}" for t in transcripts)


def _make_fail_result(reason: str, issues: list | None = None) -> dict:
    return {
        "verdict": "fail",
        "confidence": 0.0,
        "issues": issues or [{"type": "review_error", "message": reason, "evidence": None}],
        "corrections": {"summary": {}, "voc_analysis": {}, "priority_result": {}},
        "blocked_actions": [],
        "reason": reason,
    }


async def review_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]

    try:
        transcripts: list = state.get("transcripts") or []  # type: ignore[call-overload]
        analysis_result: dict = state.get("analysis_result") or {}  # type: ignore[call-overload]

        transcripts_text = _format_transcripts(transcripts)
        analysis_text = json.dumps(analysis_result, ensure_ascii=False, indent=2)

        user_msg = REVIEW_USER.format(
            transcripts=transcripts_text,
            analysis=analysis_text,
        )

        raw = await _get_caller().call_json(
            system_prompt=REVIEW_SYSTEM,
            user_message=user_msg,
            max_tokens=1000,
        )

        verdict = raw.get("verdict", "fail")
        if not ReviewVerdictValues.is_valid(verdict):
            logger.warning("review: unknown verdict=%r — fail 로 대체", verdict)
            verdict = "fail"

        review_result = {
            "verdict": verdict,
            "confidence": float(raw.get("confidence") or 0.0),
            "issues": raw.get("issues") or [],
            "corrections": raw.get("corrections") or {"summary": {}, "voc_analysis": {}, "priority_result": {}},
            "blocked_actions": raw.get("blocked_actions") or [],
            "reason": raw.get("reason") or "",
        }

        blocked_actions: list[str] = review_result["blocked_actions"]
        # retry / correctable은 human_review_required=False — graph route에서 결정
        human_review_required = verdict == "fail"

        logger.info(
            "review 완료 call_id=%s verdict=%s confidence=%.2f blocked=%s",
            call_id, verdict, review_result["confidence"], blocked_actions,
        )
        return {
            "review_result": review_result,
            "review_verdict": verdict,
            "blocked_actions": blocked_actions,
            "human_review_required": human_review_required,
        }

    except Exception as exc:
        logger.error("review 실패 call_id=%s err=%s", call_id, exc)
        fail_result = _make_fail_result(f"review_node exception: {exc}")
        return {
            "review_result": fail_result,
            "review_verdict": "fail",
            "blocked_actions": [],
            "human_review_required": True,
        }
