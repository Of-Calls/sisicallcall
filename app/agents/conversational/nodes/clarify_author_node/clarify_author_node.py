"""clarify_author_node — Option γ 재질문 생성기.

설계 배경 (Option γ):
    query_refine_node 가 결정론적 게이트에서 is_ambiguous=True 로 판정한 경우,
    intent_router_llm_node 를 완전히 우회하고 이 노드에서 재질문(clarify_question) 을
    LLM 으로 직접 생성한다.

    intent_router 는 6개 intent 분류에만 집중하도록 슬림화하고,
    Progressive Clarify 4단계 / STT 오인식 paraphrase / topic-pinned 유도질문의
    책임을 이 노드로 이전한다.

출력:
    response_text    = clarify_question  (TTS 가 그대로 읽음)
    response_path    = "clarify"         (cache_store_node 저장 차단)
    primary_intent   = "intent_clarify"
    clarify_question = clarify_question  (session_view 업데이트용)
    is_fallback      = True              (Semantic Cache 저장 추가 차단)
    is_timeout       = bool              (wait_for 초과 여부)
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

CLARIFY_AUTHOR_TIMEOUT_SEC = 3.0
MAX_TOKENS = 100
_FALLBACK_TEXT = "죄송합니다, 다시 한 번 말씀해 주시겠어요?"

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_system_prompt(tenant_name: str) -> str:
    """tenant 이름을 끼운 재질문 생성 시스템 프롬프트."""
    return f"""당신은 {tenant_name} 의 고객 상담 AI 의 재질문(clarify) 작성기입니다.
사용자 발화가 모호하다고 이미 판정된 상태입니다. 자연스러운 한국어 음성 응답으로
짧은 재질문 한 문장을 만들어내는 것이 유일한 책임입니다.

[1. Progressive 4단계 — clarify_count 별 단계적 좁힘]
입력의 clarify_count 값에 따라 질문 형태를 단계적으로 좁힙니다. available_categories
(현재 tenant 의 정제된 RAG 카테고리 목록) 을 그대로 보기로 사용합니다. 비어있을
때는 일반 옵션 ("정보 안내, 예약, 기타") 으로 fallback.

- 0 (첫 시도): open question — "어떤 도움이 필요하신가요?"
- 1: 객관식 3~4개 — available_categories 에서 3~4개 골라 보기로 만든다.
  예 (cats="메뉴 안내, 예약, 위치, 영업시간"): "메뉴 안내, 예약, 위치, 영업시간 중에 비슷한 게 있나요?"
- 2: 이분법 — 의도 구분이 큰 두 카테고리 양자택일.
  예: "정보 안내가 필요하신가요, 아니면 예약/접수가 필요하신가요?"
- 3: 단답 키워드 유도 — 2~3 단어만.
  예: "키워드 한 단어만 말씀해 주세요. '메뉴'? '예약'? '위치'?"
- 4~5: 위 4단계 후 모호하면 다른 표현 으로 다시 묻는다. 같은 단어 / 같은 카테고리 반복 금지.
- 6+ : 시스템이 자동 escalation (이 노드 호출 자체가 차단됨, query_refine 단계).

[2. Topic-pinned 질문 — rag_probe 신호 활용]
입력의 `rag_probe` 가 채워져 있고 `top_topic` 이 의미 있으면 (mid-distance 0.85~0.95 + topic 존재),
clarify_count=0 단계에서 그 topic 으로 좁힌 유도 질문을 만든다:
  - top_topic="주차 안내" → "혹시 주차 관련 문의이신가요?"
  - top_topic="영업시간" → "영업시간이 궁금하신가요?"

[3. STT 오인식 paraphrase]
refined_text 가 아래 특징을 보이면 STT 오인식 가능성:
- 실존하지 않는 기관명·지원금명·단어 조합 (예: "송년지원금", "정형외과예얀")
- 음소가 유사한 단어가 뒤섞인 표현
이런 경우 패턴: "[해석한 내용]으로 이해했는데, [추정 실제 의미]를 말씀하시는 건가요?"
  또는: "[단어]라고 말씀하신 건가요?"
표준 한국어가 명확하면 적용 금지 (정상 발화 오적용 차단).

[4. 다른 표현 강제]
last_assistant_text 와 거의 같은 문장으로 다시 묻지 말 것. 시도마다 단어·구조·카테고리 수를 바꾼다.

[5. 출력 형식 — JSON 만, 다른 텍스트 절대 금지]
{{"clarify_question": "재질문 텍스트"}}

clarify_question 은 80자 이내, 1문장. 자연스러운 음성 한국어 (사무적 표현 금지 — "확인이 어렵습니다", "정보가 없습니다" 같은 시스템 어투 사용 금지).

[6. few-shot 예시 — 4개]

입력: refined_text="음 그게..." / clarify_count=0 / available_categories="메뉴 안내, 예약"
→ {{"clarify_question": "어떤 도움이 필요하신가요?"}}

입력: refined_text="아" / clarify_count=1 / available_categories="메뉴 안내, 예약, 위치, 영업시간"
→ {{"clarify_question": "메뉴 안내, 예약, 위치, 영업시간 중에 비슷한 게 있나요?"}}

입력: refined_text="송년지원금신청" / clarify_count=0 / available_categories="청년 지원금, 보육 지원"
→ {{"clarify_question": "청년 지원금 신청 말씀하시는 건가요?"}}

입력: refined_text="모르겠어요" / clarify_count=0 / rag_probe={{"top_distance":0.88,"top_topic":"주차 안내"}}
→ {{"clarify_question": "혹시 주차 관련 문의이신가요?"}}
"""


def _build_user_message(state: CallState) -> str:
    """LLM 입력 메시지 조립 — refined_text + 컨텍스트 필드."""
    sv = state.get("session_view") or {}
    cats = state.get("available_categories") or []
    cats_line = ", ".join(cats) if cats else "(없음)"

    lines = [
        f"입력 발화: {state.get('refined_text') or state.get('normalized_text', '')}",
        f"tenant_name: {sv.get('tenant_name', '고객센터')}",
        f"clarify_count: {sv.get('clarify_count', 0)}",
        f"last_assistant_text: {sv.get('last_assistant_text')}",
        f"available_categories: {cats_line}",
    ]

    probe = state.get("rag_probe")
    if probe and probe.get("top_distance") is not None:
        matched = probe.get("matched_keywords") or []
        lines.append(
            f"rag_probe: top_distance={probe['top_distance']:.3f}"
            f" top_topic=\"{probe.get('top_topic') or ''}\""
            f" matched={matched}"
        )

    ambiguity_reason = state.get("ambiguity_reason", "")
    if ambiguity_reason:
        lines.append(f"ambiguity_reason: {ambiguity_reason}")

    return "\n".join(lines)


def _parse_response(raw: str) -> str | None:
    """LLM 응답에서 JSON 추출 → clarify_question 반환. 실패 시 None."""
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    question = parsed.get("clarify_question")
    if not question or not question.strip():
        return None
    return question.strip()


async def clarify_author_node(state: CallState) -> dict:
    """재질문 생성 노드 — query_refine_node 가 is_ambiguous=True 판정 후 진입."""
    sv = state.get("session_view") or {}
    call_id = state.get("call_id", "")
    clarify_count = sv.get("clarify_count", 0)
    refined = state.get("refined_text") or state.get("normalized_text", "")

    tenant_name = sv.get("tenant_name", "고객센터")
    system_prompt = _build_system_prompt(tenant_name)
    user_message = _build_user_message(state)

    clarify_question: str = _FALLBACK_TEXT
    is_timeout = False

    try:
        raw = await asyncio.wait_for(
            _llm.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0.1,
                max_tokens=MAX_TOKENS,
            ),
            timeout=CLARIFY_AUTHOR_TIMEOUT_SEC,
        )
        parsed = _parse_response(raw)
        if parsed:
            clarify_question = parsed
        else:
            logger.warning(
                "clarify_author parse failed call_id=%s raw=%r",
                call_id, raw[:200],
            )
    except asyncio.TimeoutError:
        logger.warning(
            "clarify_author timeout call_id=%s (%.1fs)",
            call_id, CLARIFY_AUTHOR_TIMEOUT_SEC,
        )
        is_timeout = True
    except Exception as e:
        logger.error("clarify_author error call_id=%s: %s", call_id, e)

    logger.info(
        "clarify_author call_id=%s clarify_count=%d refined=%r → question=%r",
        call_id, clarify_count, refined[:60], clarify_question[:80],
    )

    return {
        "response_text": clarify_question,
        "response_path": "clarify",
        "primary_intent": "intent_clarify",
        "clarify_question": clarify_question,
        "is_fallback": True,
        "is_timeout": is_timeout,
    }
