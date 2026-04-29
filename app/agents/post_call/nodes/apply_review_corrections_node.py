"""
Review 교정 적용 노드.

review_verdict == "correctable" 일 때 실행된다.
review_result.corrections 를 기존 분석 결과에 shallow merge 한다.
"""
from __future__ import annotations

from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)

_ALLOWED_CORRECTION_KEYS = frozenset({"summary", "voc_analysis", "priority_result"})


async def apply_review_corrections_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    review_result: dict = state.get("review_result") or {}  # type: ignore[call-overload]
    corrections: dict = review_result.get("corrections") or {}

    summary = dict(state.get("summary") or {})  # type: ignore[call-overload]
    voc_analysis = dict(state.get("voc_analysis") or {})  # type: ignore[call-overload]
    priority_result = dict(state.get("priority_result") or {})  # type: ignore[call-overload]
    analysis_result = dict(state.get("analysis_result") or {})  # type: ignore[call-overload]

    # shallow merge — 허용된 top-level key만 수정
    for key in _ALLOWED_CORRECTION_KEYS:
        if key not in corrections or not isinstance(corrections[key], dict):
            continue
        patch = corrections[key]
        if not patch:
            continue
        if key == "summary":
            summary.update(patch)
        elif key == "voc_analysis":
            voc_analysis.update(patch)
        elif key == "priority_result":
            priority_result.update(patch)

    # analysis_result도 갱신 (일관성)
    analysis_result["summary"] = summary
    analysis_result["voc_analysis"] = voc_analysis
    analysis_result["priority_result"] = priority_result

    logger.info(
        "apply_review_corrections 완료 call_id=%s corrected_keys=%s",
        call_id, [k for k in _ALLOWED_CORRECTION_KEYS if corrections.get(k)],
    )
    return {
        "summary": summary,
        "voc_analysis": voc_analysis,
        "priority_result": priority_result,
        "analysis_result": analysis_result,
    }
