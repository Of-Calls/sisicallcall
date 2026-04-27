"""Intent Router LLM Fallback — KNN 미달 / stub 시 GPT-4o-mini 로 의도 분류.

분류 결과:
    intent_faq        FAQ/매뉴얼 정보 조회
    intent_task       예약/조회/변경/취소/접수 등 업무 처리
    intent_auth       본인 확인/인증
    intent_clarify    모호한 발화 — 역질문으로 의도 재확인
    intent_escalation 상담원 연결 / AI 해결 불가
    intent_repeat     이전 AI 응답 재요청 ("다시 말해주세요", "못 들었어요" 등)

Phase 1.5 강화 (2026-04-24):
    - clarify 한도 2 → 4 (반복 시 더 자세히 묻도록)
    - Progressive Clarification (시도별 질문 전략 변경)
    - 의도 강제 추출 강화 — '거기 갈려고...' 등 단서 한 조각이라도 있으면 분류 시도
    - clarify 후속 답변에 키워드 있으면 즉시 분류 (escalation 으로 도피 금지)
    - "이전과 다르게 묻기" 강제 (last_assistant_text 활용)

Phase 2 입력 풍부화:
    - tenant_name, is_within_hours, turn_count
    - last_intent, last_question, last_assistant_text, clarify_count
"""
import asyncio
import json
import re

from app.agents.conversational.state import CallState
from app.services.llm.base import BaseLLMService
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_llm: BaseLLMService = GPT4OMiniService()

INTENT_ROUTER_TIMEOUT_SEC = 5.0  # 콜드 스타트(첫 OpenAI 호출 ~3초) 여유 포함
VALID_INTENTS = {
    "intent_faq",
    "intent_task",
    "intent_auth",
    "intent_clarify",
    "intent_escalation",
    "intent_repeat",
}

# clarify 누적 한도 — 이 횟수에 도달하면 강제 escalation.
# Phase 1.5: 2 → 4. Phase B+: 4 → 6 (최대한 대화로 해결, escalation 도피 차단).
# Progressive 전략 (라인 81~89) 4단계 + 5/6 단계는 LLM 자유 판단 (단답 키워드 반복 등).
MAX_CLARIFY_TURNS = 6


def _build_system_prompt(tenant_name: str) -> str:
    """tenant 이름을 동적으로 끼운 시스템 프롬프트 생성."""
    return f"""당신은 {tenant_name}의 고객 상담 AI 의도 분류기입니다.
고객 발화 + 직전 대화 맥락을 바탕으로 다음 5개 intent 중 정확히 하나를 선택합니다.

[intent 정의]
- intent_faq        : FAQ/매뉴얼 기반 정보 조회 (위치, 운영시간, 진료과목, 가격 등)
- intent_task       : 예약/조회/변경/취소/접수 등 업무 처리 요청
- intent_auth       : 본인 확인 / 인증 (주민번호, 회원번호 확인 등)
- intent_clarify    : 발화에 단서가 거의 없어 의도 추출이 정말 불가능할 때 — 역질문으로 재확인
- intent_escalation : 상담원 연결 요청 / AI 해결 불가 / clarify 한도 도달
- intent_repeat     : 이전 AI 응답을 다시 듣고 싶은 요청 ("다시", "못 들었어요", "다시 말해주세요", "뭐라고요" 등)

[★최우선 규칙 — 의도 강제 추출]
발화 길이로만 판단하지 말고, **단서 한 조각이라도 있으면 가능한 intent 로 분류하라.**
- "거기 갈려고", "찾아가려고", "어디예요", "위치", "주소" → intent_faq (위치)
- "진료시간", "운영시간", "몇 시까지", "언제" → intent_faq (시간)
- "예약", "잡아줘", "취소", "변경", "접수" → intent_task
- "응급", "응급실", "병상", "진료과", "주차" → intent_faq (해당 정보)
- "본인 확인", "회원번호", "주민번호" → intent_auth
- "상담원", "직원", "사람 바꿔" → intent_escalation
- intent_clarify 는 **단서가 0개**일 때만 사용 (예: "아", "음", "글쎄", "그게...", "어")

[직전이 clarify 였을 때 — 후속 답변 처리]
사용자의 답변에 키워드가 하나라도 있으면 무조건 그 키워드의 intent 로 분류한다.
- "위치", "위치안내", "어디", "주소" → intent_faq (절대 escalation 금지)
- "예약", "취소" → intent_task
- "네/응/맞아" → 직전 last_assistant_text 의 첫 번째 후보 intent
- "아니/아니요" → 두 번째 후보 intent (있다면)
- 정말로 답변이 모호하면 → intent_clarify (다음 단계 전략)

[Progressive Clarify 전략 (clarify_count 별로 다르게 질문)]
- clarify_count == 0 (첫 시도): open question, 도메인에 맞춰 자연스럽게.
  예: "어떤 도움이 필요하신가요?"
- clarify_count == 1 (두 번째 시도): **객관식 3~4개**, 카테고리로 좁힘.
  예: "위치 안내, 예약, 진료시간, 진료과목 중에 비슷한 게 있나요?"
- clarify_count == 2 (세 번째 시도): **이분법(binary)** 으로 더 단순화.
  예: "정보 안내가 필요하신가요, 아니면 예약/접수가 필요하신가요?"
- clarify_count == 3 (마지막 시도): **단답 키워드 유도**.
  예: "키워드 한 단어만 말씀해 주세요. '위치'? '예약'? '진료시간'?"

[★다른 표현 강제]
last_assistant_text 와 **거의 같은 문장으로 다시 묻지 말 것.**
시도마다 단어/구조/카테고리 수를 바꿔서 질문하라.

[복합 의도]
한 발화에 두 의도가 섞이면 primary_intent + secondary_intents 둘 다 채운다.
예: "진료시간 알려주세요 그리고 예약하고싶어요" → primary=intent_faq, secondary=[intent_task]

[영업시간 외]
is_within_hours=false 라도 **정보 조회는 intent_faq**, 업무 처리(예약 변경/취소)만 intent_escalation.

[출력 형식 — JSON 만, 다른 텍스트 절대 금지]
{{"reasoning":"한 줄 근거","primary_intent":"intent_xxx","secondary_intents":[],"routing_reason":"짧은 사유","clarify_question":null}}

[few-shot 예시]
입력 발화="진료시간이 어떻게 되나요" / last_intent=null / clarify_count=0
→ {{"reasoning":"운영시간 정보 조회","primary_intent":"intent_faq","secondary_intents":[],"routing_reason":"info_query","clarify_question":null}}

입력 발화="거기 갈려고는 어떻게 해요" / last_intent=null / clarify_count=0
→ {{"reasoning":"방문/이동 키워드, 위치 안내 의도","primary_intent":"intent_faq","secondary_intents":[],"routing_reason":"location_intent","clarify_question":null}}

입력 발화="아 병원을 찾아가려고 해요" / last_intent=null / clarify_count=0
→ {{"reasoning":"병원 찾아가기 = 위치/길 안내","primary_intent":"intent_faq","secondary_intents":[],"routing_reason":"location_intent","clarify_question":null}}

입력 발화="예약 하고싶어요" / last_intent=null / clarify_count=0
→ {{"reasoning":"예약 업무 요청","primary_intent":"intent_task","secondary_intents":[],"routing_reason":"task_request","clarify_question":null}}

입력 발화="아" / last_intent=null / clarify_count=0
→ {{"reasoning":"단서 0개, 의도 추출 불가","primary_intent":"intent_clarify","secondary_intents":[],"routing_reason":"no_signal","clarify_question":"어떤 도움이 필요하신가요?"}}

입력 발화="위치안내요" / last_intent=intent_clarify / clarify_count=2
→ {{"reasoning":"clarify 후속에 명확 키워드 '위치안내', 즉시 FAQ","primary_intent":"intent_faq","secondary_intents":[],"routing_reason":"clarify_followup_keyword","clarify_question":null}}

입력 발화="네" / last_intent=intent_clarify / last_assistant_text="진료 안내를 드릴까요?" / clarify_count=1
→ {{"reasoning":"역질문 긍정, 진료 안내 채택","primary_intent":"intent_faq","secondary_intents":[],"routing_reason":"clarify_followup_yes","clarify_question":null}}

입력 발화="아니요" / last_intent=intent_clarify / last_assistant_text="진료 안내를 드릴까요? 예약을 도와드릴까요?" / clarify_count=1
→ {{"reasoning":"진료 거부, 두 번째 후보(예약) 채택","primary_intent":"intent_task","secondary_intents":[],"routing_reason":"clarify_followup_no","clarify_question":null}}

입력 발화="음 그게..." / last_intent=intent_clarify / last_assistant_text="어떤 도움이 필요하신가요?" / clarify_count=1
→ {{"reasoning":"여전히 단서 없음, 객관식으로 좁히기","primary_intent":"intent_clarify","secondary_intents":[],"routing_reason":"narrow_choices","clarify_question":"위치 안내, 예약, 진료시간, 진료과목 중에 비슷한 게 있나요?"}}

입력 발화="잘 모르겠네" / last_intent=intent_clarify / last_assistant_text="위치 안내, 예약, 진료시간, 진료과목 중에 비슷한 게 있나요?" / clarify_count=2
→ {{"reasoning":"객관식에도 답 못함, 이분법으로 단순화","primary_intent":"intent_clarify","secondary_intents":[],"routing_reason":"binary_choice","clarify_question":"정보 안내가 필요하신가요, 아니면 예약/접수가 필요하신가요?"}}

입력 발화="음..." / last_intent=intent_clarify / clarify_count=3
→ {{"reasoning":"마지막 시도, 단답 키워드 유도","primary_intent":"intent_clarify","secondary_intents":[],"routing_reason":"keyword_prompt","clarify_question":"키워드 한 단어만 말씀해 주세요. '위치'? '예약'? '진료시간'?"}}

입력 발화="상담원 좀 바꿔주세요" / last_intent=null
→ {{"reasoning":"명시적 상담원 요청","primary_intent":"intent_escalation","secondary_intents":[],"routing_reason":"explicit_escalation","clarify_question":null}}

입력 발화="진료시간 알려주고 예약도 하고싶어요" / last_intent=null
→ {{"reasoning":"FAQ + Task 복합 의도","primary_intent":"intent_faq","secondary_intents":["intent_task"],"routing_reason":"compound_query","clarify_question":null}}

입력 발화="제 회원번호로 확인해주세요" / last_intent=null
→ {{"reasoning":"본인 확인 요청","primary_intent":"intent_auth","secondary_intents":[],"routing_reason":"identity_check","clarify_question":null}}

입력 발화="다시 한번 말해주세요" / last_assistant_text="평일 외래 진료는 09:00~17:30입니다."
→ {{"reasoning":"이전 AI 응답 재요청","primary_intent":"intent_repeat","secondary_intents":[],"routing_reason":"repeat_request","clarify_question":null}}

입력 발화="뭐라고요" / last_assistant_text="역삼역 3번 출구에서 도보 5분 거리입니다."
→ {{"reasoning":"잘 못 들었다는 재요청","primary_intent":"intent_repeat","secondary_intents":[],"routing_reason":"repeat_request","clarify_question":null}}

입력 발화="다시 말씀해 주세요" / last_assistant_text=null
→ {{"reasoning":"이전 응답 없는 재요청, FAQ 처리","primary_intent":"intent_faq","secondary_intents":[],"routing_reason":"no_prev_response","clarify_question":null}}
"""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_user_message(state: CallState) -> str:
    """Phase 2 — session_view 의 맥락 정보를 모두 LLM 입력으로 풍부화.

    barge-in turn (직전 AI 응답이 사용자 발화로 끊긴 경우) 에는 끊긴 응답 원문을
    추가 라인으로 끼워 넣어, 새 발화가 끊긴 응답에 대한 후속/거부/재요청인지
    완전히 새로운 의도인지 LLM 이 판단하도록 한다.
    """
    sv = state.get("session_view") or {}
    knn_intent = state.get("knn_intent")
    base = (
        f"입력 발화: {state['normalized_text']}\n"
        f"tenant_name: {sv.get('tenant_name', '고객센터')}\n"
        f"is_within_hours: {sv.get('is_within_hours', True)}\n"
        f"turn_count: {sv.get('turn_count', 0)}\n"
        f"last_intent: {sv.get('last_intent')}\n"
        f"last_question: {sv.get('last_question')}\n"
        f"last_assistant_text: {sv.get('last_assistant_text')}\n"
        f"clarify_count: {sv.get('clarify_count', 0)}\n"
        f"KNN 후보 intent: {knn_intent or '없음'}"
    )
    interrupted = (state.get("interrupted_response_text") or "").strip()
    if state.get("is_bargein") and interrupted:
        base += (
            f"\nbarge-in: 직전 AI 응답이 사용자 발화로 중단됨. "
            f"끊긴 응답 원문='{interrupted[:200]}'. "
            f"새 발화가 (a) 끊긴 응답에 대한 후속/거부/재요청 인지, "
            f"(b) 완전히 새로운 의도인지 판단해 분류하라."
        )
    return base


def _parse_intent_response(raw: str) -> dict | None:
    """LLM 응답에서 JSON 추출 → primary_intent 검증.

    intent_clarify 인 경우 clarify_question 도 같이 추출. 없거나 비어있으면 안전망 문구로 대체.
    실패 시 None 반환.
    """
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

    clarify_question = parsed.get("clarify_question")
    if intent == "intent_clarify" and not clarify_question:
        clarify_question = "조금 더 자세히 말씀해 주시겠어요?"

    out = {
        "primary_intent": intent,
        "secondary_intents": parsed.get("secondary_intents") or [],
        "routing_reason": parsed.get("routing_reason") or "llm_routed",
    }
    if intent == "intent_clarify":
        out["clarify_question"] = clarify_question
    return out


def _fallback(knn_intent: str | None, reason: str) -> dict:
    """LLM timeout/parse 실패 시 폴백 분류.

    KNN 후보가 있으면 그쪽으로, 없으면 clarify 로 (escalation 아님).
    이유: 의도 파악 자체 실패 = "다시 한 번 물어보는" 게 자연스러움.
    escalation 은 "AI 로 정말 못 풀 때" 만 (clarify 한도 도달, 명시 요청 등).
    """
    if knn_intent in VALID_INTENTS:
        return {
            "primary_intent": knn_intent,
            "secondary_intents": [],
            "routing_reason": reason,
        }
    return {
        "primary_intent": "intent_clarify",
        "secondary_intents": [],
        "routing_reason": f"{reason}_clarify_fallback",
        "clarify_question": "죄송합니다, 다시 한 번 말씀해 주시겠어요?",
    }


def _force_escalation_if_clarify_exhausted(state: CallState) -> dict | None:
    """clarify 가 누적 한도(4) 에 도달한 경우 LLM 호출 전 즉시 escalation.

    LLM 결과가 또 clarify 일 위험 차단 + 비용/지연 절약. Progressive 전략에서
    4번째 시도(keyword_prompt) 후에도 여전히 모호하면 사람에게 넘긴다.
    """
    sv = state.get("session_view") or {}
    if sv.get("clarify_count", 0) >= MAX_CLARIFY_TURNS:
        logger.info(
            "intent_router clarify 누적 한도 초과 call_id=%s clarify_count=%d → 강제 escalation",
            state["call_id"], sv.get("clarify_count"),
        )
        return {
            "primary_intent": "intent_escalation",
            "secondary_intents": [],
            "routing_reason": "clarify_exhausted_forced",
        }
    return None


async def intent_router_llm_node(state: CallState) -> dict:
    forced = _force_escalation_if_clarify_exhausted(state)
    if forced is not None:
        return forced

    sv = state.get("session_view") or {}
    tenant_name = sv.get("tenant_name", "고객센터")
    knn_intent = state.get("knn_intent")
    user_message = _build_user_message(state)

    try:
        raw = await asyncio.wait_for(
            _llm.generate(
                system_prompt=_build_system_prompt(tenant_name),
                user_message=user_message,
                temperature=0.1,
                max_tokens=300,
            ),
            timeout=INTENT_ROUTER_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning("intent_router timeout call_id=%s", state["call_id"])
        return _fallback(knn_intent, "knn_fallback_timeout")
    except Exception as e:
        logger.error("intent_router error call_id=%s: %s", state["call_id"], e)
        return _fallback(knn_intent, "knn_fallback_error")

    parsed = _parse_intent_response(raw)
    if parsed is None:
        logger.warning(
            "intent_router parse failed call_id=%s raw=%r",
            state["call_id"], raw[:200],
        )
        return _fallback(knn_intent, "knn_fallback_parse_error")

    logger.info(
        "intent_router 결과 call_id=%s intent=%s reason=%s clarify=%s",
        state["call_id"], parsed["primary_intent"], parsed["routing_reason"],
        bool(parsed.get("clarify_question")),
    )
    return parsed
