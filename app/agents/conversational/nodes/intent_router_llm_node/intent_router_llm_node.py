"""Intent Router LLM — Cache miss 후 GPT-4o-mini 로 의도 분류.

분류 결과 (6개):
    intent_faq        FAQ/매뉴얼 정보 조회
    intent_task       예약/조회/변경/취소/접수 등 업무 처리
    intent_auth       본인 확인/인증
    intent_clarify    의도가 불분명해 재질문이 필요한 경우
    intent_escalation 상담원 연결 명시 요청 / clarify 한도 도달 / AI 해결 불가
    intent_repeat     이전 AI 응답 재요청 ("다시", "못 들었어요" 등)

핵심 정책 (2026-04-28 재설계):
    - 명확 키워드 발화 → 해당 intent 직접 분류
    - 모호 발화 → intent_clarify 로 재질문 (추측 분류 금지)
    - Progressive 4단계 (open → 객관식 → 이분법 → 단답) 으로 점진적 의도 좁힘
    - MAX_CLARIFY_TURNS=6 도달 시 LLM 호출 전 자동 escalation
    - 영업시간 외 처리는 escalation_branch_node 가 결정적 처리 — 본 노드 무관

입력 컨텍스트 (session_view 경유):
    - tenant_name, turn_count, last_intent, last_question, last_assistant_text, clarify_count
    - is_bargein + interrupted_response_text (barge-in 발생 시)
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

# clarify 누적 한도 — 이 횟수에 도달하면 LLM 호출 전 강제 escalation.
# Progressive 전략 4단계 (open/객관식/이분법/단답) + 5~6 단계는 LLM 자유 판단 (다른 표현 반복).
# 최대한 대화로 해결, escalation 도피 차단 목적.
MAX_CLARIFY_TURNS = 6


def _build_system_prompt(tenant_name: str) -> str:
    """tenant 이름을 동적으로 끼운 시스템 프롬프트 생성."""
    return f"""당신은 {tenant_name}의 고객 상담 AI 의도 분류기입니다.
고객 발화 + 직전 대화 맥락을 바탕으로 다음 6개 intent 중 정확히 하나를 선택합니다.

[1. intent 정의]
- intent_faq        : FAQ/매뉴얼 정보 조회 (위치, 운영시간, 진료과목, 가격 등)
- intent_task       : 예약/조회/변경/취소/접수 등 업무 처리 요청
- intent_auth       : 본인 확인 / 인증 (회원번호, 주민번호 확인 등)
- intent_clarify    : 의도가 불분명해 재질문이 필요한 경우
- intent_escalation : 상담원 연결 명시 요청 / clarify 한도 도달 / AI 해결 불가
- intent_repeat     : 이전 AI 응답을 다시 듣고 싶은 요청 ("다시", "못 들었어요" 등)

[2. ★기본 분류 원칙 — 추측 금지]
- 발화에 명확한 키워드 / 동작 표현이 있으면 해당 intent 로 분류한다.
- 키워드가 모호하거나 단서가 부족하면 → intent_clarify 로 재질문한다.
- **추측 분류 금지.** 확실하지 않으면 사용자에게 묻는 게 더 도움이 된다.
- 잘못된 강제 분류는 RAG 검색 실패로 이어져 통화 흐름을 망친다.

[3. 명확 분류 신호 (직접 키워드)]
- 위치 / 주소 / 길 / 오시는 길 / 찾아가다 → intent_faq
- 진료시간 / 운영시간 / 영업시간 / 몇 시까지 / 몇 시 → intent_faq
- 진료과 / 진료과목 / 응급실 / 주차 / 주차장 → intent_faq
- 예약 / 접수 / 취소 / 변경 / 조회 → intent_task
- 본인확인 / 회원번호 / 주민번호 / 인증 → intent_auth
- 상담원 / 직원 / 사람 / 사람으로 / 바꿔주세요 → intent_escalation
- 다시 / 다시 한번 / 뭐라고요 / 못 들었어요 → intent_repeat

(주의: "언제" 단독은 모호 — task/faq 양쪽 가능. 다른 키워드와 함께 등장할 때만 분류)

[4. clarify 가 정답인 경우]
- 단답 / 비언어 표현: "아", "음", "글쎄", "그게...", "어"
- 동사·키워드 없이 정황만: "거기 갈려고", "그게 뭐예요", "어떻게 해요"
- 정보만 나열: "내일 오후에", "55세 여자입니다"
- 의도가 둘 이상으로 갈리는 모호한 표현

[5. 직전이 clarify 였을 때 — 후속 답변 처리]
사용자가 clarify 질문에 답한 turn 에서는 다음 우선순위로 분류한다:
1. 답변에 [3] 의 명확 키워드가 있으면 즉시 그 intent (escalation 으로 도피 금지)
2. 답변이 "네/맞아/응" 이고 last_assistant_text 가 **이분법** ("정보 안내인가요, 예약인가요?") 이면 첫 번째 후보의 intent
3. 답변이 "아니요/아니" 이고 last_assistant_text 가 **이분법** 이면 두 번째 후보의 intent
4. 위 어디에도 안 맞으면 → intent_clarify (다음 단계 Progressive)

(객관식 4개 같은 비-이분법에서 "네" 만 오면 의미 없음 → 그냥 clarify 다음 단계)

[6. Progressive Clarify 전략 — clarify_count 별]
입력의 `clarify_count` 값에 따라 질문 형태를 단계적으로 좁힌다.
- 0 (첫 시도): open question, 자연스럽게.
  예: "어떤 도움이 필요하신가요?"
- 1: **객관식 3~4개**, 도메인 카테고리로 좁힘.
  예: "위치 안내, 예약, 진료시간, 진료과목 중에 비슷한 게 있나요?"
- 2: **이분법 (binary)** 으로 단순화.
  예: "정보 안내가 필요하신가요, 아니면 예약/접수가 필요하신가요?"
- 3: **단답 키워드 유도**.
  예: "키워드 한 단어만 말씀해 주세요. '위치'? '예약'? '진료시간'?"
- 4~5: 위 4단계 후에도 모호하면 **이전과 다른 표현** 으로 다시 묻는다.
  같은 카테고리 / 같은 단어 반복 금지. 더 짧게, 더 구체적으로.
- 6 도달 시 시스템이 자동 escalation — LLM 호출 자체가 차단됨.

[7. ★다른 표현 강제]
last_assistant_text 와 거의 같은 문장으로 다시 묻지 말 것.
시도마다 단어·구조·카테고리 수를 바꿔 질문한다.

[8. 복합 의도]
한 발화에 두 의도가 섞이면 primary_intent + secondary_intents 둘 다 채운다.
예: "진료시간 알려주세요 그리고 예약하고싶어요" → primary=intent_faq, secondary=[intent_task]

[9. barge-in 처리]
입력에 `barge-in:` 메타가 포함되면 직전 AI 응답이 사용자 발화로 끊긴 상황이다.
끊긴 응답 원문(`interrupted_response_text`) 과 새 발화의 관계를 판단한다:
- 끊긴 응답에 대한 후속/거부/재요청 → repeat 또는 그 응답이 유도한 intent
- 완전히 새로운 의도 → 그 의도로 분류

[10. 출력 형식 — JSON 만, 다른 텍스트 절대 금지]
{{"reasoning":"한 줄 근거","primary_intent":"intent_xxx","secondary_intents":[],"routing_reason":"짧은 사유","clarify_question":null}}

intent_clarify 일 때 clarify_question 필드에 [6] 전략에 맞는 질문 텍스트 채우기 (필수).
다른 intent 일 때 null.

[11. few-shot 예시 — 7개]

입력 발화="진료시간이 어떻게 되나요" / last_intent=null / clarify_count=0
→ {{"reasoning":"운영시간 정보 조회 키워드","primary_intent":"intent_faq","secondary_intents":[],"routing_reason":"info_query","clarify_question":null}}

입력 발화="예약 좀 잡아주세요" / last_intent=null / clarify_count=0
→ {{"reasoning":"예약 업무 요청","primary_intent":"intent_task","secondary_intents":[],"routing_reason":"task_request","clarify_question":null}}

입력 발화="상담원 좀 바꿔주세요" / last_intent=null / clarify_count=0
→ {{"reasoning":"명시적 상담원 요청","primary_intent":"intent_escalation","secondary_intents":[],"routing_reason":"explicit_escalation","clarify_question":null}}

입력 발화="다시 한번 말해주세요" / last_assistant_text="평일 외래 진료는 09:00~17:30입니다." / clarify_count=0
→ {{"reasoning":"이전 AI 응답 재요청","primary_intent":"intent_repeat","secondary_intents":[],"routing_reason":"repeat_request","clarify_question":null}}

입력 발화="거기 갈려고는 어떻게 해요" / last_intent=null / clarify_count=0
→ {{"reasoning":"명확 키워드 부재, 의도 추측 금지","primary_intent":"intent_clarify","secondary_intents":[],"routing_reason":"ambiguous","clarify_question":"어떤 부분이 궁금하신가요?"}}

입력 발화="위치안내요" / last_intent=intent_clarify / clarify_count=2
→ {{"reasoning":"clarify 후속에 명확 키워드 '위치안내', 즉시 FAQ","primary_intent":"intent_faq","secondary_intents":[],"routing_reason":"clarify_followup_keyword","clarify_question":null}}

입력 발화="음 그게..." / last_intent=intent_clarify / last_assistant_text="어떤 도움이 필요하신가요?" / clarify_count=1
→ {{"reasoning":"여전히 단서 없음, 객관식으로 좁히기","primary_intent":"intent_clarify","secondary_intents":[],"routing_reason":"narrow_choices","clarify_question":"위치 안내, 예약, 진료시간, 진료과목 중에 비슷한 게 있나요?"}}
"""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_user_message(state: CallState) -> str:
    """Phase 2 — session_view 의 맥락 정보를 모두 LLM 입력으로 풍부화.

    barge-in turn (직전 AI 응답이 사용자 발화로 끊긴 경우) 에는 끊긴 응답 원문을
    추가 라인으로 끼워 넣어, 새 발화가 끊긴 응답에 대한 후속/거부/재요청인지
    완전히 새로운 의도인지 LLM 이 판단하도록 한다.
    """
    sv = state.get("session_view") or {}
    base = (
        f"입력 발화: {state['normalized_text']}\n"
        f"tenant_name: {sv.get('tenant_name', '고객센터')}\n"
        f"is_within_hours: {sv.get('is_within_hours', True)}\n"
        f"turn_count: {sv.get('turn_count', 0)}\n"
        f"last_intent: {sv.get('last_intent')}\n"
        f"last_question: {sv.get('last_question')}\n"
        f"last_assistant_text: {sv.get('last_assistant_text')}\n"
        f"clarify_count: {sv.get('clarify_count', 0)}"
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


def _fallback(reason: str) -> dict:
    """LLM timeout/parse 실패 시 폴백 분류 — 항상 clarify 로 회귀.

    의도 파악 자체 실패 = "다시 한 번 물어보는" 게 자연스러움.
    escalation 은 "AI 로 정말 못 풀 때" 만 (clarify 한도 도달, 명시 요청 등).
    """
    return {
        "primary_intent": "intent_clarify",
        "secondary_intents": [],
        "routing_reason": f"{reason}_clarify_fallback",
        "clarify_question": "죄송합니다, 다시 한 번 말씀해 주시겠어요?",
    }


def _force_escalation_if_clarify_exhausted(state: CallState) -> dict | None:
    """clarify 가 누적 한도(MAX_CLARIFY_TURNS=6) 에 도달한 경우 LLM 호출 전 즉시 escalation.

    LLM 결과가 또 clarify 일 위험 차단 + 비용/지연 절약. Progressive 전략 4단계
    + 다른 표현 2회까지 시도해도 여전히 모호하면 사람에게 넘긴다.
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
        return _fallback("llm_timeout")
    except Exception as e:
        logger.error("intent_router error call_id=%s: %s", state["call_id"], e)
        return _fallback("llm_error")

    parsed = _parse_intent_response(raw)
    if parsed is None:
        logger.warning(
            "intent_router parse failed call_id=%s raw=%r",
            state["call_id"], raw[:200],
        )
        return _fallback("llm_parse_error")

    logger.info(
        "intent_router 결과 call_id=%s intent=%s reason=%s clarify=%s",
        state["call_id"], parsed["primary_intent"], parsed["routing_reason"],
        bool(parsed.get("clarify_question")),
    )
    return parsed
