"""
시연용 Post-call 컨텍스트 픽스처.

demo context 특성:
- call_id = demo-call-critical
- customer_phone 포함
- angry + escalated → Slack, SMS, Notion, Gmail, CompanyDB 액션 유도
- callback 필요 문구 포함 → Calendar, SMS 액션 유도
- critical priority → Slack, SMS, Notion 추가 유도
- POST_CALL_ENABLE_NOTION_RECORD → Notion call record 유도
"""
from __future__ import annotations

DEMO_POST_CALL_CONTEXT: dict = {
    "metadata": {
        "call_id": "demo-call-critical",
        "tenant_id": "demo-tenant",
        "customer_phone": "+821049460829",
        "start_time": "2026-04-28T10:00:00Z",
        "end_time": "2026-04-28T10:12:00Z",
        "status": "completed",
    },
    "transcripts": [
        {
            "role": "customer",
            "text": "지금 당장 해결 안 되면 소비자원에 신고하겠습니다. 정말 너무 화가 납니다!",
            "timestamp": "2026-04-28T10:01:00Z",
        },
        {
            "role": "agent",
            "text": "불편 드려서 진심으로 죄송합니다. 즉시 담당팀에 에스컬레이션 처리하겠습니다.",
            "timestamp": "2026-04-28T10:02:00Z",
        },
        {
            "role": "customer",
            "text": "다음 주 월요일 오전에 콜백 전화 꼭 받아야 해요. 콜백 꼭 해주세요.",
            "timestamp": "2026-04-28T10:04:00Z",
        },
        {
            "role": "agent",
            "text": "네, 다음 주 월요일 오전으로 콜백 예약 처리해드리겠습니다.",
            "timestamp": "2026-04-28T10:05:00Z",
        },
    ],
    "branch_stats": {"faq": 0, "task": 0, "escalation": 1},
}

# 시연용 LLM mock 반환값 — angry + escalated + critical + callback
DEMO_LLM_SUMMARY = {
    "summary_short": "[시연] 고객 긴급 에스컬레이션 — 즉시 대응 필요",
    "summary_detailed": "[시연] 고객이 서비스 불만으로 극도로 화를 내며 에스컬레이션 요청. 콜백 요청함.",
    "customer_intent": "불만 처리 및 콜백 예약",
    "customer_emotion": "angry",
    "resolution_status": "escalated",
    "keywords": ["불만", "에스컬레이션", "콜백", "소비자원"],
    "handoff_notes": "고객이 콜백을 강하게 요청함. 월요일 오전 콜백 필요.",
}

DEMO_LLM_VOC = {
    "sentiment_result": {"sentiment": "angry", "intensity": 0.95, "reason": "극도로 화남"},
    "intent_result": {
        "primary_category": "서비스 불만",
        "sub_categories": ["에스컬레이션", "콜백 요청"],
        "is_repeat_topic": False,
        "faq_candidate": False,
    },
    "priority_result": {
        "priority": "critical",
        "action_required": True,
        "suggested_action": "즉시 에스컬레이션 후 콜백 예약",
        "reason": "angry 감정 + escalated 상태 → 즉시 대응 필요",
    },
}

DEMO_LLM_PRIORITY = {
    "priority": "critical",
    "tier": "critical",
    "action_required": True,
    "suggested_action": "즉시 에스컬레이션 후 콜백 예약",
    "reason": "angry + escalated + critical — 즉시 조치",
}
