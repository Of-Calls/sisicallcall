"""Intent Router LLM — Cache miss + query_refine 통과 후 GPT-4o-mini 로 의도 분류.

2026-04-30 Option γ 리팩터로 책임 축소:
    - 정규화 / 띄어쓰기 정정 → query_refine_node (룰 기반, deterministic)
    - 모호성 판정 / clarify 발사 → query_refine_node (gate) + clarify_author_node (LLM)
    - Progressive 4단계 / STT 오인식 paraphrase → clarify_author_node
    - MAX_CLARIFY_TURNS 강제 escalation → query_refine_node

본 노드는 *명확하다고 판정된 발화* 만 받아 5개 intent 중 하나로 분류:
    intent_faq        FAQ/매뉴얼 정보 조회
    intent_task       예약/조회/변경/취소/접수 등 업무 처리
    intent_auth       본인 확인/인증
    intent_escalation 상담원 연결 명시 요청 / AI 해결 불가
    intent_repeat     이전 AI 응답 재요청

intent_clarify 는 *본 노드에서 출력하지 않음* — 모호성은 이미 upstream 에서 결정.

성능 목표: prompt ~1400 토큰 (이전 ~3500 토큰 대비 60% 축소), hardcut 2.5s
(이전 5.0s). server_120020.log Turn 1 timeout 사고 (4985ms) 재발 방지.
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

INTENT_ROUTER_TIMEOUT_SEC = 2.5  # slim prompt 라 cold 도 2.5s 안에 들어옴
VALID_INTENTS = {
    "intent_faq",
    "intent_task",
    "intent_auth",
    "intent_escalation",
    "intent_repeat",
}


def _build_system_prompt(tenant_name: str) -> str:
    """tenant 이름을 동적으로 끼운 시스템 프롬프트 생성 (slim, ~1400 토큰)."""
    return f"""당신은 {tenant_name}의 고객 상담 AI 의도 분류기입니다.
입력 발화는 이미 *명확하다고 판정된* 상태입니다 (모호 발화는 upstream 에서 차단됨).
다음 5개 intent 중 정확히 하나를 선택합니다.

[1. intent 정의]
- intent_faq        : FAQ/매뉴얼 정보 조회 (위치, 운영시간, 진료과목, 가격 등)
- intent_task       : 예약/조회/변경/취소/접수 등 업무 처리 요청
- intent_auth       : 본인 확인 / 인증 (회원번호, 주민번호 확인 등)
- intent_escalation : 상담원 연결 명시 요청 / AI 해결 불가
- intent_repeat     : 이전 AI 응답을 다시 듣고 싶은 요청 ("다시", "못 들었어요" 등)

[2. ★최우선 인증 규칙 — 모든 키워드 분류보다 상위]
다음 조건에 해당하면 intent_auth 로 분류:
1. **auth_pending == true** → intent_auth (이전 턴 인증 확인 발화 후 사용자 응답 대기)
2. **rag_probe.is_auth == true AND matched_keywords 1개 이상** → intent_auth
   - is_auth=true 이어도 matched_keywords 가 비어있으면 발동하지 않음
   - 예) "예약하고싶어요" → is_auth=true + matched=['예약'] → intent_auth 우선
   - 예) "영업시간이어떻게되세요" → is_auth=true 청크가 top-1 이어도 matched=[] 이면 [3]로 넘김

[3. 명확 분류 신호 (직접 키워드)]
- 위치 / 주소 / 길 / 오시는 길 / 찾아가다 / 찾아가 / 가려고 / 어디로 / 어떻게 가 / 가는 법 → intent_faq
- 진료시간 / 운영시간 / 영업시간 / 몇 시까지 / 몇 시 → intent_faq
- 진료과 / 진료과목 / 응급실 / 주차 / 주차장 → intent_faq
- 예약 / 접수 / 취소 / 변경 / 조회 → intent_task
- 본인확인 / 회원번호 / 주민번호 / 인증 → intent_auth
- 상담원 / 직원 / 사람 / 사람으로 / 바꿔주세요 → intent_escalation
- 다시 / 다시 한번 / 뭐라고요 / 못 들었어요 → intent_repeat

★중요: 입력은 query_refine 가 이미 띄어쓰기 정정 + 필러 제거를 마친 상태지만, 만약을
대비해 한 덩어리 문자열 ("구청을찾아가려고요") 에서도 위 키워드의 부분 문자열을
찾으면 그대로 적용한다. (주의: "언제" 단독은 모호 — 다른 키워드와 함께 등장할 때만)

[4. RAG 신호 활용 — 모호 발화 보강]
입력의 `rag_probe` 가 채워져 있으면 (cache miss 직후 RAG top_k=3 검색 신호) 활용한다.
[2] 인증 미해당 + [3] 의 명확 키워드 부재 케이스에 한해 적용. rag_probe 가 null 이면 무시.

- top_distance ≤ 0.70 → **FAQ 매우 강신호**.
  matched_keywords 가 비어있어도 (STT 띄어쓰기 부재 케이스 포함) intent_faq 로
  자신 있게 분류한다. 거리 자체가 임베딩이 답을 가지고 있음을 보장.
- top_distance ≤ 0.85 + matched_keywords 1개 이상 → **FAQ 강신호**.
  intent_faq 로 분류 (RAG 가 답할 수 있다는 신호).
- 그 외 (약신호 / 무신호) → 키워드 매핑이 부재해도 발화 의도를 *추측해서라도* 5개
  intent 중 가장 가까운 것으로 분류. (모호 케이스는 upstream 에서 이미 차단됨.)

[5. barge-in 처리]
입력에 `barge-in:` 메타가 포함되면 직전 AI 응답이 사용자 발화로 끊긴 상황이다.
끊긴 응답 원문(`interrupted_response_text`) 과 새 발화의 관계를 판단한다:
- 끊긴 응답에 대한 후속/거부/재요청 → repeat 또는 그 응답이 유도한 intent
- 완전히 새로운 의도 → 그 의도로 분류

[6. 출력 형식 — JSON 만, 다른 텍스트 절대 금지]
{{"reasoning":"한 줄 근거","primary_intent":"intent_xxx"}}

primary_intent 는 위 5개 중 하나여야 한다 (intent_clarify 는 출력 금지 — upstream
에서만 결정됨). 다른 값이면 자동 escalation 처리됨.

[7. few-shot 예시 — 7개]

입력 발화="진료시간이 어떻게 되나요" / last_intent=null
→ {{"reasoning":"운영시간 정보 조회 키워드","primary_intent":"intent_faq"}}

입력 발화="예약 좀 잡아주세요" / last_intent=null / rag_probe={{"top_distance":0.72,"matched_keywords":["예약"],"is_auth":true}}
→ {{"reasoning":"[2] 인증 우선 — is_auth=true + matched=['예약']","primary_intent":"intent_auth"}}

입력 발화="상담원 좀 바꿔주세요" / last_intent=null
→ {{"reasoning":"명시적 상담원 요청","primary_intent":"intent_escalation"}}

입력 발화="다시 한번 말해주세요" / last_assistant_text="평일 외래 진료는 09:00~17:30입니다."
→ {{"reasoning":"이전 AI 응답 재요청","primary_intent":"intent_repeat"}}

입력 발화="강남구청을찾아가려고요" / rag_probe={{"top_distance":0.49,"matched_keywords":[],"top_topic":"위치","is_auth":false}}
→ {{"reasoning":"위치 질문 — '찾아가' 부분 문자열 + rag_probe 매우 강신호(0.49)","primary_intent":"intent_faq"}}

입력 발화="영업시간이어떻게되세요" / rag_probe={{"top_distance":0.65,"matched_keywords":[],"top_topic":"예약","is_auth":true}}
→ {{"reasoning":"[2] is_auth=true 이지만 matched=[] → 미발동. [3] 영업시간 키워드 → faq","primary_intent":"intent_faq"}}

입력 발화="네 맞아요" / last_assistant_text="인증 진행하시겠습니까?" / auth_pending=true
→ {{"reasoning":"auth_pending=true — [2]-1 최우선 적용","primary_intent":"intent_auth"}}
"""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_user_message(state: CallState) -> str:
    """slim user message — refined_text + 핵심 컨텍스트만.

    clarify_count / Progressive 관련 필드는 query_refine / clarify_author 가 처리하므로
    본 노드 입력에서 제거.
    """
    sv = state.get("session_view") or {}
    text = state.get("refined_text") or state.get("normalized_text") or ""
    base = (
        f"입력 발화: {text}\n"
        f"tenant_name: {sv.get('tenant_name', '고객센터')}\n"
        f"last_intent: {sv.get('last_intent')}\n"
        f"last_assistant_text: {sv.get('last_assistant_text')}"
    )
    probe = state.get("rag_probe")
    if probe and probe.get("top_distance") is not None:
        matched = probe.get("matched_keywords") or []
        base += (
            f"\nrag_probe:"
            f"\n  top_distance: {probe['top_distance']:.3f}"
            f"\n  matched_keywords: {', '.join(matched) if matched else '(없음)'}"
            f"\n  top_topic: {probe.get('top_topic') or '(없음)'}"
            f"\n  is_auth: {probe.get('is_auth', False)}"
        )
    base += f"\nauth_pending: {state.get('auth_pending', False)}"
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

    intent_clarify 가 출력되면 invalid 처리 (refine/clarify_author 가 결정해야 하는 영역).
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

    return {"primary_intent": intent}


def _fallback() -> dict:
    """LLM timeout/parse 실패 시 폴백 — escalation.

    이전엔 intent_clarify 로 회귀했지만, query_refine 가 이미 *명확하다* 고 판정한
    발화이므로 다시 clarify 로 돌리는 건 무의미. 사람에게 넘기는 게 맞음.
    """
    return {"primary_intent": "intent_escalation"}


async def intent_router_llm_node(state: CallState) -> dict:
    sv = state.get("session_view") or {}
    tenant_name = sv.get("tenant_name", "고객센터")
    user_message = _build_user_message(state)

    try:
        raw = await asyncio.wait_for(
            _llm.generate(
                system_prompt=_build_system_prompt(tenant_name),
                user_message=user_message,
                temperature=0.1,
                max_tokens=100,
            ),
            timeout=INTENT_ROUTER_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning("intent_router timeout call_id=%s", state["call_id"])
        return _fallback()
    except Exception as e:
        logger.error("intent_router error call_id=%s: %s", state["call_id"], e)
        return _fallback()

    parsed = _parse_intent_response(raw)
    if parsed is None:
        logger.warning(
            "intent_router parse failed call_id=%s raw=%r",
            state["call_id"], raw[:200],
        )
        return _fallback()

    logger.info(
        "intent_router 결과 call_id=%s intent=%s",
        state["call_id"], parsed["primary_intent"],
    )
    return parsed
