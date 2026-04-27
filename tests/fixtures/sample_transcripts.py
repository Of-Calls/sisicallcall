"""테스트용 샘플 transcript 및 post-call payload 픽스처."""
from __future__ import annotations

# ── Case 1: resolved + neutral ────────────────────────────────────────────────

TRANSCRIPTS_RESOLVED_NEUTRAL = [
    {"role": "customer", "text": "요금제 변경하고 싶은데요."},
    {"role": "agent", "text": "네, 도와드리겠습니다. 어떤 요금제로 변경을 원하시나요?"},
    {"role": "customer", "text": "더 저렴한 걸로 바꾸고 싶어요."},
    {"role": "agent", "text": "5G 라이트 요금제를 추천드립니다. 변경해 드릴까요?"},
    {"role": "customer", "text": "네, 그렇게 해주세요. 감사합니다."},
]

PAYLOAD_RESOLVED_NEUTRAL = {
    "call_id": "call-resolved-001",
    "tenant_id": "tenant-a",
    "trigger": "call_ended",
    "summary": {
        "summary_short": "요금제 변경 완료",
        "summary_detailed": "고객이 5G 라이트 요금제로 변경 요청 후 즉시 처리됨",
        "customer_intent": "요금제 변경",
        "customer_emotion": "neutral",
        "resolution_status": "resolved",
        "keywords": ["요금제", "변경", "5G"],
        "handoff_notes": None,
    },
    "voc_analysis": {
        "sentiment_result": {"sentiment": "neutral", "intensity": 0.1, "reason": "만족함"},
        "intent_result": {
            "primary_category": "요금 문의",
            "sub_categories": ["요금제 변경"],
            "is_repeat_topic": False,
            "faq_candidate": True,
        },
        "priority_result": {
            "priority": "low",
            "action_required": False,
            "suggested_action": None,
            "reason": "정상 처리 완료",
        },
    },
    "priority_result": {
        "priority": "low",
        "tier": "low",
        "action_required": False,
        "suggested_action": None,
        "reason": "정상 처리 완료",
    },
    "action_plan": {
        "action_required": False,
        "actions": [],
        "rationale": "우선순위 낮음, 액션 불필요",
    },
    "executed_actions": [],
    "errors": [],
    "partial_success": False,
}


# ── Case 2: angry + escalated + critical ─────────────────────────────────────

TRANSCRIPTS_ANGRY_CRITICAL = [
    {"role": "customer", "text": "또 오류가 났잖아요! 이게 몇 번째예요?"},
    {"role": "agent", "text": "죄송합니다. 확인해 보겠습니다."},
    {"role": "customer", "text": "매번 이런 식이면 해지할 거예요!"},
    {"role": "agent", "text": "불편을 드려 대단히 죄송합니다. 즉시 팀장에게 에스컬레이션하겠습니다."},
    {"role": "customer", "text": "빨리 해결해 주세요."},
]

PAYLOAD_ANGRY_CRITICAL = {
    "call_id": "call-angry-002",
    "tenant_id": "tenant-a",
    "trigger": "escalation_immediate",
    "summary": {
        "summary_short": "반복 오류 고객 에스컬레이션",
        "summary_detailed": "고객이 반복적인 서비스 오류로 강한 불만을 표출하고 해지를 언급함",
        "customer_intent": "오류 해결 요구",
        "customer_emotion": "angry",
        "resolution_status": "escalated",
        "keywords": ["오류", "해지", "에스컬레이션"],
        "handoff_notes": "고객 반복 불만, 팀장 승인 필요",
    },
    "voc_analysis": {
        "sentiment_result": {"sentiment": "angry", "intensity": 0.9, "reason": "반복 오류 강한 불만"},
        "intent_result": {
            "primary_category": "서비스 오류",
            "sub_categories": ["반복 민원", "해지 위협"],
            "is_repeat_topic": True,
            "faq_candidate": False,
        },
        "priority_result": {
            "priority": "critical",
            "action_required": True,
            "suggested_action": "즉시 팀장 에스컬레이션 + VOC 등록",
            "reason": "고객 분노 및 해지 위협",
        },
    },
    "priority_result": {
        "priority": "critical",
        "tier": "critical",
        "action_required": True,
        "suggested_action": "즉시 팀장 에스컬레이션 + VOC 등록",
        "reason": "고객 분노 및 해지 위협",
    },
    "action_plan": {
        "action_required": True,
        "actions": [
            {
                "action_type": "create_voc_issue",
                "tool": "company_db",
                "priority": "critical",
                "params": {
                    "tier": "critical",
                    "priority": "critical",
                    "primary_category": "서비스 오류",
                    "reason": "고객 분노 및 해지 위협",
                    "summary_short": "반복 오류 고객 에스컬레이션",
                },
                "status": "success",
                "result": {"created": True, "issue_id": "VOC-MOCK-call-angry-002"},
                "external_id": "VOC-MOCK-call-angry-002",
                "error": None,
            },
            {
                "action_type": "send_manager_email",
                "tool": "gmail",
                "priority": "critical",
                "params": {
                    "to": "manager@example.com",
                    "subject": "[긴급] 고객 에스컬레이션",
                    "body": "고객 분노 및 해지 위협 상황",
                },
                "status": "success",
                "result": {"sent": True},
                "external_id": "gmail-mock-call-angry-002",
                "error": None,
            },
        ],
        "rationale": "critical 우선순위, 즉각 조치 필요",
    },
    "executed_actions": [
        {
            "action_type": "create_voc_issue",
            "tool": "company_db",
            "status": "success",
            "external_id": "VOC-MOCK-call-angry-002",
            "error": None,
            "result": {"created": True, "issue_id": "VOC-MOCK-call-angry-002"},
            "params": {"tier": "critical"},
        },
        {
            "action_type": "send_manager_email",
            "tool": "gmail",
            "status": "success",
            "external_id": "gmail-mock-call-angry-002",
            "error": None,
            "result": {"sent": True},
            "params": {"to": "manager@example.com", "subject": "[긴급] 고객 에스컬레이션"},
        },
    ],
    "errors": [],
    "partial_success": False,
}


# ── Case 3: negative + repeated issue + action_required ──────────────────────

TRANSCRIPTS_NEGATIVE_REPEATED = [
    {"role": "customer", "text": "저번에도 이 문제로 전화했는데 또 발생했어요."},
    {"role": "agent", "text": "죄송합니다. 이전 이력을 확인하겠습니다."},
    {"role": "customer", "text": "빠른 처리 부탁드립니다. 불편하네요."},
    {"role": "agent", "text": "확인 결과 이전 처리가 완전하지 않았습니다. 재처리 예약해 드리겠습니다."},
]

PAYLOAD_NEGATIVE_REPEATED = {
    "call_id": "call-negative-003",
    "tenant_id": "tenant-b",
    "trigger": "call_ended",
    "summary": {
        "summary_short": "반복 문의 재처리 예약",
        "summary_detailed": "동일 문제 재발로 고객이 재문의, 재처리 예약으로 부분 해결",
        "customer_intent": "반복 문제 해결",
        "customer_emotion": "negative",
        "resolution_status": "resolved",
        "keywords": ["반복", "재처리", "불편"],
        "handoff_notes": "이전 이력 확인 필요",
    },
    "voc_analysis": {
        "sentiment_result": {"sentiment": "negative", "intensity": 0.6, "reason": "반복 발생 불만"},
        "intent_result": {
            "primary_category": "서비스 품질",
            "sub_categories": ["반복 민원"],
            "is_repeat_topic": True,
            "faq_candidate": False,
        },
        "priority_result": {
            "priority": "high",
            "action_required": True,
            "suggested_action": "VOC 등록 및 콜백 예약",
            "reason": "반복 문의 고객",
        },
    },
    "priority_result": {
        "priority": "high",
        "tier": "high",
        "action_required": True,
        "suggested_action": "VOC 등록 및 콜백 예약",
        "reason": "반복 문의 고객",
    },
    "action_plan": {
        "action_required": True,
        "actions": [
            {
                "action_type": "create_voc_issue",
                "tool": "company_db",
                "priority": "high",
                "params": {
                    "tier": "high",
                    "priority": "high",
                    "primary_category": "서비스 품질",
                    "reason": "반복 문의 고객",
                    "summary_short": "반복 문의 재처리 예약",
                },
                "status": "success",
                "result": {"created": True, "issue_id": "VOC-MOCK-call-negative-003"},
                "external_id": "VOC-MOCK-call-negative-003",
                "error": None,
            },
            {
                "action_type": "schedule_callback",
                "tool": "calendar",
                "priority": "high",
                "params": {"title": "반복 문의 고객 재콜백"},
                "status": "success",
                "result": {"scheduled": True, "event_id": "calendar-mock-call-negative-003"},
                "external_id": "calendar-mock-call-negative-003",
                "error": None,
            },
        ],
        "rationale": "반복 문의, VOC 등록 및 콜백 예약 필요",
    },
    "executed_actions": [
        {
            "action_type": "create_voc_issue",
            "tool": "company_db",
            "status": "success",
            "external_id": "VOC-MOCK-call-negative-003",
            "error": None,
            "result": {"created": True, "issue_id": "VOC-MOCK-call-negative-003"},
            "params": {"tier": "high"},
        },
        {
            "action_type": "schedule_callback",
            "tool": "calendar",
            "status": "failed",
            "external_id": None,
            "error": "calendar MCP timeout",
            "result": {},
            "params": {"title": "반복 문의 고객 재콜백"},
        },
    ],
    "errors": [],
    "partial_success": True,
}


# ── 공통 접근 목록 ─────────────────────────────────────────────────────────────

ALL_SAMPLE_PAYLOADS = [
    PAYLOAD_RESOLVED_NEUTRAL,
    PAYLOAD_ANGRY_CRITICAL,
    PAYLOAD_NEGATIVE_REPEATED,
]

ALL_SAMPLE_TRANSCRIPTS = [
    TRANSCRIPTS_RESOLVED_NEUTRAL,
    TRANSCRIPTS_ANGRY_CRITICAL,
    TRANSCRIPTS_NEGATIVE_REPEATED,
]
