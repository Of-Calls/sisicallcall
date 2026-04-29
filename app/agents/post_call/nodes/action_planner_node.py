from __future__ import annotations

import os

from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── 콜백 필요성 판단 키워드 ────────────────────────────────────────────────────
_CALLBACK_KEYWORDS: tuple[str, ...] = ("콜백", "callback", "연락", "다시 전화")


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _is_callback_needed(handoff_notes: str, suggested_action: str) -> bool:
    combined = f"{handoff_notes} {suggested_action}".lower()
    return any(kw.lower() in combined for kw in _CALLBACK_KEYWORDS)


def _build_plan(
    call_id: str,
    tenant_id: str,
    summary: dict,
    voc: dict,
    priority: dict,
    customer_phone: str = "",
) -> dict:
    """rule-based action plan 생성기.

    MCP 를 직접 호출하지 않으며, action_type / tool / params / status 만 결정한다.
    실제 실행은 action_router → ActionExecutor 가 담당한다.
    """

    # ── 신호 추출 ─────────────────────────────────────────────────────────────
    emotion: str = summary.get("customer_emotion", "neutral")
    resolution: str = summary.get("resolution_status", "resolved")
    handoff_notes: str = summary.get("handoff_notes") or ""
    summary_short: str = summary.get("summary_short", "")

    sentiment_r: dict = voc.get("sentiment_result") or {}
    intent_r: dict = voc.get("intent_result") or {}
    voc_priority_r: dict = voc.get("priority_result") or {}

    sentiment: str = sentiment_r.get("sentiment", "neutral")
    is_repeat: bool = bool(intent_r.get("is_repeat_topic", False))
    faq_candidate: bool = bool(intent_r.get("faq_candidate", False))
    primary_category: str = intent_r.get("primary_category", "")
    voc_action_required: bool = bool(voc_priority_r.get("action_required", False))

    priority_level: str = priority.get("priority", "low")
    action_required: bool = bool(priority.get("action_required", False)) or voc_action_required
    suggested_action: str = priority.get("suggested_action") or ""
    priority_reason: str = priority.get("reason", "")

    # ── Rule 1: early exit ────────────────────────────────────────────────────
    if not action_required and not faq_candidate:
        return {
            "action_required": False,
            "actions": [],
            "rationale": "액션 불필요: action_required=False, faq_candidate=False",
        }

    # ── 액션 누적 (action_type 기준 dedup) ────────────────────────────────────
    # dict 삽입 순서 보장(Python 3.7+)으로 dedup 처리
    action_set: dict[str, dict] = {}
    rationale_parts: list[str] = []

    base_params: dict = {
        "call_id": call_id,
        "tenant_id": tenant_id,
        "priority": priority_level,
        "summary_short": summary_short,
        "primary_category": primary_category,
        "reason": priority_reason,
    }

    def _add(action_type: str, tool: str, extra_params: dict, why: str) -> None:
        """중복 action_type 은 첫 번째 등록만 유지한다 (Rule 8)."""
        if action_type not in action_set:
            action_set[action_type] = {
                "action_type": action_type,
                "tool": tool,
                "priority": priority_level,
                "params": {**base_params, **extra_params},
                "status": "pending",
            }
            rationale_parts.append(why)

    # ── Rule 3: angry + (escalated | abandoned) ───────────────────────────────
    is_angry: bool = emotion == "angry" or sentiment == "angry"
    is_unresolved: bool = resolution in ("escalated", "abandoned")
    if is_angry and is_unresolved:
        _add("create_voc_issue", "company_db", {},
             "angry+에스컬레이션 → VOC 등록")
        _add("send_manager_email", "gmail",
            {"subject": f"[긴급 에스컬레이션] {call_id}"},
            "angry+에스컬레이션 → 팀장 이메일")
        _add("add_priority_queue", "internal_dashboard", {},
             "angry+에스컬레이션 → 우선순위 큐")
        _add("send_slack_alert", "slack",
            {"channel_type": "critical", "message": f"[긴급] {call_id}: {summary_short}"},
            "angry+에스컬레이션 → Slack 긴급 알림")
        _add("send_voc_receipt_sms", "sms",
             {"customer_phone": customer_phone},
             "angry+에스컬레이션 → SMS VOC 접수 안내")

    # ── Rule 4: negative + repeated issue ─────────────────────────────────────
    is_negative: bool = sentiment in ("negative", "angry") or emotion in ("negative", "angry")
    if is_negative and is_repeat:
        _add("create_voc_issue", "company_db", {},
             "negative+반복 문의 → VOC 등록")

    # ── Rule 5: critical priority → 필수 액션 ─────────────────────────────────
    if priority_level == "critical":
        _add("send_manager_email", "gmail",
            {"subject": f"[CRITICAL] {call_id}"},
            "critical priority → 팀장 이메일 필수")
        _add("send_slack_alert", "slack",
            {"channel_type": "critical", "message": f"[CRITICAL] {call_id}: {summary_short}"},
            "critical priority → Slack 알림 필수")
        _add("send_voc_receipt_sms", "sms",
             {"customer_phone": customer_phone},
             "critical priority → SMS VOC 접수 안내")

    # ── Rule 6: 콜백 필요성 감지 ──────────────────────────────────────────────
    if _is_callback_needed(handoff_notes, suggested_action):
        _add("schedule_callback", "calendar",
             {"callback_reason": suggested_action or "에스컬레이션 후 콜백"},
             "콜백 필요 감지 → 콜백 예약")
        _add("send_callback_sms", "sms",
             {"customer_phone": customer_phone},
             "콜백 필요 감지 → SMS 콜백 안내")

    # ── Rule 7: faq_candidate ─────────────────────────────────────────────────
    # Rule 2 ("action_required=false여도 faq_candidate=true이면 생성 가능") 도 여기서 처리
    if faq_candidate:
        _add("mark_faq_candidate", "internal_dashboard",
             {"question": primary_category},
             "faq_candidate=True → FAQ 후보 등록")

    # ── Rule J1: JIRA_MCP_REAL=true → Jira 이슈 생성 ────────────────────────────
    if os.getenv("JIRA_MCP_REAL", "").lower() in ("1", "true"):
        jira_triggered = (
            (priority_level in ("high", "critical") and action_required)
            or (is_angry and is_unresolved)
        )
        if jira_triggered:
            _add(
                "create_jira_issue",
                "jira",
                {
                    "summary": f"[{priority_level.upper()}] {summary_short or call_id}",
                    "description": priority_reason or summary_short or suggested_action or "",
                    "labels": ["sisicallcall", "post-call", priority_level],
                },
                "JIRA_MCP_REAL=true → Jira 이슈 생성",
            )

    # ── Rule N4: POST_CALL_ENABLE_NOTION_RECORD=true → 통화 1건 = Notion row 1개 ─
    if os.getenv("POST_CALL_ENABLE_NOTION_RECORD", "").lower() in ("1", "true"):
        _add("create_notion_call_record", "notion",
             {
                 "customer_phone": customer_phone,
                 "customer_emotion": emotion,
                 "resolution_status": resolution,
                 "action_required": action_required,
             },
             "POST_CALL_ENABLE_NOTION_RECORD=true → Notion 통화 기록")

    # ── 기본 폴백: high/critical + action_required 이면 VOC 등록 보장 ──────────
    if action_required and priority_level in ("high", "critical") and "create_voc_issue" not in action_set:
        _add("create_voc_issue", "company_db", {},
             f"{priority_level} priority + action_required → 기본 VOC 등록")

    # ── 최후 폴백: action_required 이지만 아무 액션도 없을 때 ─────────────────
    if action_required and not action_set:
        _add("create_voc_issue", "company_db", {},
             "action_required=True → 기본 VOC 등록")

    actions = list(action_set.values())
    rationale = "; ".join(rationale_parts) if rationale_parts else "규칙 기반 액션 없음"

    return {
        "action_required": len(actions) > 0,
        "actions": actions,
        "rationale": rationale,
    }


# ── 노드 엔트리포인트 ─────────────────────────────────────────────────────────

async def action_planner_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    try:
        summary: dict = state.get("summary") or {}        # type: ignore[call-overload]
        voc: dict = state.get("voc_analysis") or {}       # type: ignore[call-overload]
        priority: dict = state.get("priority_result") or {}  # type: ignore[call-overload]
        metadata: dict = state.get("call_metadata") or {}  # type: ignore[call-overload]
        customer_phone: str = metadata.get("customer_phone", "")

        plan = _build_plan(
            call_id=call_id,
            tenant_id=state["tenant_id"],
            summary=summary,
            voc=voc,
            priority=priority,
            customer_phone=customer_phone,
        )
        logger.info(
            "action_planner 완료 call_id=%s actions=%d action_required=%s",
            call_id, len(plan["actions"]), plan["action_required"],
        )
        return {"action_plan": plan}

    except Exception as exc:
        logger.error("action_planner 실패 call_id=%s err=%s", call_id, exc)
        errors = list(state.get("errors", []))  # type: ignore[call-overload]
        errors.append({"node": "action_planner", "error": str(exc)})
        return {"action_plan": None, "errors": errors, "partial_success": True}
