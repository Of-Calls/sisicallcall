"""
시연용 Post-call 컨텍스트 픽스처.

demo context 특성:
- call_id = demo-call-critical
- customer_phone 포함
- angry + escalated → Slack, SMS, Notion, Gmail, CompanyDB 액션 유도
- callback 필요 문구 포함 → Calendar, SMS 액션 유도
- critical priority → Slack, SMS, Notion 추가 유도
- JIRA_MCP_REAL → Jira 이슈 생성 유도
- POST_CALL_ENABLE_NOTION_RECORD → Notion call record 유도
"""
from __future__ import annotations

DEMO_POST_CALL_CONTEXT: dict = {
    "metadata": {
        "call_id": "demo-call-critical",
        "tenant_id": "demo-tenant",
        "customer_phone": "+821049460829",
        "start_time": "2026-04-28T14:00:00Z",
        "end_time": "2026-04-28T14:14:00Z",
        "status": "completed",
    },
    "transcripts": [
        {
            "role": "customer",
            "text": "환불 요청한 지 2주가 지났는데 아직 입금이 안 됐습니다. 지난주에도 전화했는데 또 처음부터 설명해야 하나요?",
            "timestamp": "2026-04-28T14:01:00Z",
        },
        {
            "role": "agent",
            "text": "반복해서 불편을 드려 죄송합니다. 결제/환불 담당팀에 긴급 건으로 바로 에스컬레이션하겠습니다.",
            "timestamp": "2026-04-28T14:03:00Z",
        },
        {
            "role": "customer",
            "text": "오늘 안에 담당자가 직접 전화 주세요. 처리 상황도 문자로 꼭 안내해 주세요. 더는 기다리기 어렵습니다.",
            "timestamp": "2026-04-28T14:06:00Z",
        },
        {
            "role": "agent",
            "text": "네, 오늘 중 담당자 콜백 요청을 남기고 환불 지연 건으로 긴급 후속 안내 문자를 발송하겠습니다.",
            "timestamp": "2026-04-28T14:08:00Z",
        },
    ],
    "branch_stats": {"faq": 0, "task": 1, "escalation": 1},
}

# 시연용 LLM mock 반환값 — angry + escalated + critical + callback
DEMO_LLM_SUMMARY = {
    "summary_short": "[시연] 환불 지연 반복 문의 — 긴급 후속 조치 필요",
    "summary_detailed": "[시연] 고객이 환불 지연으로 반복 문의를 했으며 강한 불만을 표현함. 결제/환불 담당팀 에스컬레이션과 오늘 중 담당자 콜백, 문자 안내를 요청함.",
    "customer_intent": "환불 지연 확인 및 담당자 콜백 요청",
    "customer_emotion": "angry",
    "resolution_status": "escalated",
    "keywords": ["환불 지연", "반복 문의", "콜백", "긴급 후속", "문자 안내"],
    "handoff_notes": "환불 지연 반복 문의 건. 고객이 오늘 중 담당자 콜백과 처리 상황 문자 안내를 요청함.",
}

DEMO_LLM_VOC = {
    "sentiment_result": {
        "sentiment": "angry",
        "intensity": 0.93,
        "reason": "환불 지연과 반복 문의로 강한 불만을 표현함",
    },
    "intent_result": {
        "primary_category": "환불/결제",
        "sub_categories": ["환불 지연", "반복 문의", "담당자 콜백", "문자 안내"],
        "is_repeat_topic": True,
        "faq_candidate": False,
    },
    "priority_result": {
        "priority": "critical",
        "action_required": True,
        "suggested_action": "환불 담당팀 에스컬레이션 후 오늘 중 콜백",
        "reason": "반복 문의 + 환불 지연 + angry 감정 → 긴급 후속 조치 필요",
    },
}

DEMO_LLM_PRIORITY = {
    "priority": "critical",
    "tier": "critical",
    "action_required": True,
    "suggested_action": "환불 담당팀 에스컬레이션 후 오늘 중 콜백",
    "reason": "환불 지연 반복 문의와 강한 불만으로 긴급 후속 조치 필요",
}

# ── 통합 분석 mock (post_call_analysis_node 전용) ─────────────────────────────
# DEMO_LLM_SUMMARY / DEMO_LLM_VOC / DEMO_LLM_PRIORITY 는 호환용으로 유지한다.
DEMO_LLM_ANALYSIS = {
    "summary": DEMO_LLM_SUMMARY,
    "voc_analysis": DEMO_LLM_VOC,
    "priority_result": DEMO_LLM_PRIORITY,
}

# ── Review Pass mock (review_node 전용) ──────────────────────────────────────
DEMO_LLM_REVIEW_PASS = {
    "verdict": "pass",
    "confidence": 0.95,
    "issues": [],
    "corrections": {
        "summary": {},
        "voc_analysis": {},
        "priority_result": {},
    },
    "blocked_actions": [],
    "reason": "Transcript supports the analysis.",
}
