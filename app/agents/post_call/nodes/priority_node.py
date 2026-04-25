from __future__ import annotations

import json

from app.agents.post_call.llm_caller import PostCallLLMCaller, make_priority_caller
from app.agents.post_call.prompts import PRIORITY_SYSTEM, PRIORITY_USER
from app.agents.post_call.schemas import PriorityLevel, PriorityNodeResult
from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)

_caller: PostCallLLMCaller | None = None


def _get_caller() -> PostCallLLMCaller:
    global _caller
    if _caller is None:
        _caller = make_priority_caller()
    return _caller


def _validate(raw: dict) -> dict:
    """Pydantic 검증 + enum fallback.

    action_planner_node 가 priority.get("tier") 를 참조하므로
    tier 는 반드시 priority 와 동일한 값으로 채워야 한다.
    """
    priority_val = raw.get("priority", "low")
    if priority_val not in PriorityLevel._value2member_map_:
        logger.warning("priority: unknown priority=%r — low 로 대체", priority_val)
        priority_val = "low"

    # tier 를 priority 와 동기화 (action_planner 하위 호환)
    raw["priority"] = priority_val
    raw["tier"] = priority_val

    result = PriorityNodeResult(**raw)
    return result.model_dump()


async def priority_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    try:
        summary = state.get("summary") or {}  # type: ignore[call-overload]
        voc = state.get("voc_analysis") or {}  # type: ignore[call-overload]

        summary_text = summary.get("summary_detailed") or summary.get("summary_short") or "(요약 없음)"
        user_msg = PRIORITY_USER.format(
            summary=summary_text,
            voc_analysis=json.dumps(voc, ensure_ascii=False, indent=2),
        )

        raw = await _get_caller().call_json(
            system_prompt=PRIORITY_SYSTEM,
            user_message=user_msg,
            max_tokens=512,
        )
        priority = _validate(raw)
        logger.info(
            "priority 완료 call_id=%s priority=%s action_required=%s",
            call_id,
            priority.get("priority"),
            priority.get("action_required"),
        )
        return {"priority_result": priority}

    except Exception as exc:
        logger.error("priority 실패 call_id=%s err=%s", call_id, exc)
        errors = list(state.get("errors", []))  # type: ignore[call-overload]
        errors.append({"node": "priority", "error": str(exc)})
        return {"priority_result": None, "errors": errors, "partial_success": True}
