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

성능 목표: prompt ~1600 토큰 (STEP A→B→C 구조), hardcut 2.5s.
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

INTENT_ROUTER_TIMEOUT_SEC = 2.5
VALID_INTENTS = {
    "intent_faq",
    "intent_task",
    "intent_auth",
    "intent_escalation",
    "intent_repeat",
}


def _build_system_prompt(tenant_name: str) -> str:
    """STEP A→B→C 구조 시스템 프롬프트. auth gate 를 실행 전 단계로 격리."""
    return f"""당신은 {tenant_name}의 고객 상담 AI 의도 분류기입니다.
입력 발화는 이미 *명확하다고 판정된* 상태입니다.
다음 5개 intent 중 하나로 분류합니다.

[1. intent 정의]
- intent_faq        : FAQ/매뉴얼 정보 조회 (위치, 운영시간, 진료과목, 가격 등)
- intent_task       : 예약/조회/변경/취소/접수 등 업무 처리 요청
- intent_auth       : 본인 확인 / 인증 (회원번호, 주민번호 확인 등)
- intent_escalation : 상담원 연결 명시 요청 / AI 해결 불가
- intent_repeat     : 이전 AI 응답을 다시 듣고 싶은 요청 ("다시", "못 들었어요" 등)

[STEP A — 인증 게이트 (반드시 가장 먼저 확인. 해당하면 즉시 intent_auth 출력하고 B/C 평가 금지)]
  A1. auth_pending == true → intent_auth 확정
  A2. 입력에 [AUTH_GATE_PRECHECK] A2_should_fire=YES 가 명시된 경우 → intent_auth 확정
      (= rag_probe.is_auth=true AND matched_keywords 1개 이상)
      ★ 발화에 "예약", "접수", "변경" 같은 업무 키워드가 포함되어 있어도 A2 가 우선한다.
        이 키워드들은 인증 청크가 top-1 인 경우 인증 흐름의 시작 발화이기 때문이다.

  A1/A2 모두 해당 없음 → STEP B 로 진행.

[STEP B — 직접 키워드 분류 (STEP A 미해당 시에만 평가)]
- 위치/주소/길/오시는 길/찾아가/가려고/어디로/어떻게 가/가는 법 → intent_faq
- 진료시간/운영시간/영업시간/몇 시까지/몇 시 → intent_faq
- 진료과/진료과목/응급실/주차/주차장 → intent_faq
- 예약/접수/취소/변경/조회/신청 → intent_task  (단, STEP A2 해당 시 여기 오지 않음)
- 상담원/직원/사람/사람으로/바꿔주세요 → intent_escalation
- 다시/다시 한번/뭐라고요/못 들었어요 → intent_repeat
한 덩어리 문자열("구청을찾아가려고요")에서도 위 키워드 부분 문자열을 찾으면 적용.
("언제" 단독은 모호 — 다른 키워드와 함께 등장할 때만 적용.)

[STEP C — RAG 신호 활용 (STEP A/B 미해당 시에만)]
rag_probe 가 null 이면 발화 의도를 추측해 5개 intent 중 선택.
null 이 아니면:
- top_distance ≤ 0.70 → FAQ 매우 강신호. matched_keywords 없어도 intent_faq.
- top_distance ≤ 0.85 + matched_keywords 1개 이상 → FAQ 강신호. intent_faq.
- 그 외 → 발화 의도 추측해 5개 중 가장 가까운 것으로 분류.

[barge-in 처리]
입력에 `barge-in:` 메타가 포함되면 직전 AI 응답이 끊긴 상황이다.
새 발화가 끊긴 응답 후속/거부/재요청이면 repeat 또는 그 intent, 새 의도면 그 intent.

[출력 형식 — JSON 만, 다른 텍스트 절대 금지]
{{"auth_gate":"A1_match"|"A2_match"|"no_match","step_used":"A"|"B"|"C","reasoning":"한 줄 근거","primary_intent":"intent_xxx"}}
primary_intent 는 위 5개 중 하나 (intent_clarify 출력 금지).

[few-shot 예시]

[A1 — auth_pending]
입력="네 맞아요" / auth_pending=true / last_assistant="인증 진행하시겠습니까?"
→ {{"auth_gate":"A1_match","step_used":"A","reasoning":"auth_pending=true — A1 즉시 적중.","primary_intent":"intent_auth"}}

[A2 — '예약' 함정 케이스 1]
입력="예약은 가능한가요" / [AUTH_GATE_PRECHECK] A2_should_fire=YES / rag_probe={{top_distance:0.669,matched:["예약"],is_auth:true}}
→ {{"auth_gate":"A2_match","step_used":"A","reasoning":"A2_should_fire=YES → A2 적중. STEP B '예약→intent_task' 무시. 인증 청크가 top-1.","primary_intent":"intent_auth"}}

[A2 — '예약' 함정 케이스 2]
입력="예약하고 싶어요" / [AUTH_GATE_PRECHECK] A2_should_fire=YES / rag_probe={{top_distance:0.71,matched:["예약"],is_auth:true}}
→ {{"auth_gate":"A2_match","step_used":"A","reasoning":"A2_should_fire=YES → A2 적중.","primary_intent":"intent_auth"}}

[A 미통과 — is_auth=true 이지만 matched=[]]
입력="영업시간이어떻게되세요" / [AUTH_GATE_PRECHECK] A2_should_fire=NO / rag_probe={{top_distance:0.65,matched:[],is_auth:true}}
→ {{"auth_gate":"no_match","step_used":"B","reasoning":"A2_should_fire=NO(matched=[]) → A 미통과. STEP B '영업시간' → intent_faq.","primary_intent":"intent_faq"}}

[B — 일반 FAQ]
입력="진료시간이 어떻게 되나요" / [AUTH_GATE_PRECHECK] A2_should_fire=NO
→ {{"auth_gate":"no_match","step_used":"B","reasoning":"A 미통과. STEP B '진료시간' → intent_faq.","primary_intent":"intent_faq"}}

[B — escalation]
입력="상담원 좀 바꿔주세요" / [AUTH_GATE_PRECHECK] A2_should_fire=NO
→ {{"auth_gate":"no_match","step_used":"B","reasoning":"A 미통과. STEP B '상담원/바꿔주세요' → intent_escalation.","primary_intent":"intent_escalation"}}

[C — RAG 매우 강신호]
입력="강남구청을찾아가려고요" / [AUTH_GATE_PRECHECK] A2_should_fire=NO / rag_probe={{top_distance:0.49,matched:[],top_topic:"위치",is_auth:false}}
→ {{"auth_gate":"no_match","step_used":"C","reasoning":"A/B 미통과. STEP C: top_distance=0.49 ≤ 0.70 → FAQ 매우 강신호.","primary_intent":"intent_faq"}}

[B — repeat]
입력="다시 한번 말해주세요" / [AUTH_GATE_PRECHECK] A2_should_fire=NO / last_assistant="평일 외래 진료는 09:00~17:30입니다."
→ {{"auth_gate":"no_match","step_used":"B","reasoning":"A 미통과. STEP B '다시 한번' → intent_repeat.","primary_intent":"intent_repeat"}}
"""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_user_message(state: CallState) -> str:
    """slim user message — refined_text + 핵심 컨텍스트 + AUTH_GATE_PRECHECK 사전계산."""
    sv = state.get("session_view") or {}
    text = state.get("refined_text") or state.get("normalized_text") or ""
    base = (
        f"입력 발화: {text}\n"
        f"tenant_name: {sv.get('tenant_name', '고객센터')}\n"
        f"auth_pending: {state.get('auth_pending', False)}"
    )
    if sv.get("last_intent"):
        base += f"\nlast_intent: {sv['last_intent']}"
    base += f"\nlast_assistant_text: {sv.get('last_assistant_text')}"

    probe = state.get("rag_probe")
    if probe and probe.get("top_distance") is not None:
        matched = probe.get("matched_keywords") or []
        is_auth = probe.get("is_auth", False)
        auth_hit = is_auth and bool(matched)
        base += (
            f"\n[AUTH_GATE_PRECHECK] is_auth={is_auth} matched_count={len(matched)}"
            f" → A2_should_fire={'YES' if auth_hit else 'NO'}"
            f"\nrag_probe:"
            f"\n  top_distance: {probe['top_distance']:.3f}"
            f"\n  matched_keywords: {', '.join(matched) if matched else '(없음)'}"
            f"\n  top_topic: {probe.get('top_topic') or '(없음)'}"
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


def _parse_intent_response(raw: str, state: CallState) -> dict | None:
    """LLM 응답에서 JSON 추출 → primary_intent 검증 + 결정론적 auth gate 오버라이드.

    LLM 이 STEP A 를 무시하고 intent_task 를 반환해도 코드가 강제 보정.
    override 발생 시 WARNING 로그 → LLM 프롬프트 준수율 측정 지표로 활용.
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

    # AUTH GATE 강제 — LLM 이 무시해도 코드가 잡는다
    if state.get("auth_pending"):
        if intent != "intent_auth":
            logger.warning(
                "intent_router auth_pending override call_id=%s llm=%s → intent_auth",
                state["call_id"], intent,
            )
        return {"primary_intent": "intent_auth"}

    probe = state.get("rag_probe") or {}
    if probe.get("is_auth") and (probe.get("matched_keywords") or []):
        if intent != "intent_auth":
            logger.warning(
                "intent_router rag_auth override call_id=%s llm=%s matched=%s → intent_auth",
                state["call_id"], intent, probe.get("matched_keywords"),
            )
        return {"primary_intent": "intent_auth"}

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

    parsed = _parse_intent_response(raw, state)
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
