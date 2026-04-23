import asyncio
import json
import re

from app.agents.conversational.state import CallState
from app.services.llm.base import BaseLLMService
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.utils.logger import get_logger

# KNN low-confidence fallback — GPT-4o-mini 1.5초 하드컷 (feature_spec.md §5)
# 출력: primary_intent, secondary_intents, routing_reason
# 타임아웃/파싱 실패 시 → knn_intent 폴백, knn_intent도 없으면 intent_escalation

logger = get_logger(__name__)

_llm: BaseLLMService = GPT4OMiniService()

# 명세 초안은 1.5s 였으나 한국 ↔ OpenAI 네트워크 왕복(200~300ms) + 응답(0.5~2s) 합산 시
# 매번 timeout 위험. 실통화 검증(2026-04-23) 결과 3.0s 가 현실적 값으로 확정.
# 추후 OpenAI 클라이언트 connection keep-alive 도입 시 재검토.
INTENT_ROUTER_TIMEOUT_SEC = 3.0
VALID_INTENTS = {"intent_faq", "intent_task", "intent_auth", "intent_escalation"}

# TODO(agents.md 이관): 담당자 배정 후 프롬프트를 agents.md 로 이관하고 여기서는 import
INTENT_ROUTER_SYSTEM_PROMPT = """당신은 고객센터 AI의 의도 분류기입니다.
고객 발화를 아래 4개 intent 중 정확히 하나로 분류하고, JSON만 반환하세요.
- intent_faq: FAQ/매뉴얼 기반 정보 조회 질문
- intent_task: 예약/조회/변경/취소/접수 등 업무 처리 요청
- intent_auth: 본인 확인/인증 관련 요청
- intent_escalation: 상담원 연결 요청 또는 AI로 해결 불가 상황

출력 형식 (JSON만, 다른 텍스트 금지):
{"primary_intent": "intent_xxx", "secondary_intents": [], "routing_reason": "짧은 근거"}"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_user_message(normalized_text: str, knn_intent: str | None, session_view: dict) -> str:
    turn_count = session_view.get("turn_count", 0) if session_view else 0
    return (
        f"고객 발화: {normalized_text}\n"
        f"KNN 후보 intent: {knn_intent or '없음'}\n"
        f"현재 턴 번호: {turn_count}"
    )


def _parse_intent_response(raw: str) -> dict | None:
    """LLM 응답에서 JSON 추출 → primary_intent 검증. 실패 시 None."""
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

    return {
        "primary_intent": intent,
        "secondary_intents": parsed.get("secondary_intents") or [],
        "routing_reason": parsed.get("routing_reason") or "llm_routed",
    }


def _fallback(knn_intent: str | None, reason: str) -> dict:
    if knn_intent in VALID_INTENTS:
        return {
            "primary_intent": knn_intent,
            "secondary_intents": [],
            "routing_reason": reason,
        }
    return {
        "primary_intent": "intent_escalation",
        "secondary_intents": [],
        "routing_reason": f"{reason}_no_knn",
    }


async def intent_router_llm_node(state: CallState) -> dict:
    knn_intent = state.get("knn_intent")
    user_message = _build_user_message(
        state["normalized_text"], knn_intent, state.get("session_view") or {}
    )

    try:
        raw = await asyncio.wait_for(
            _llm.generate(
                system_prompt=INTENT_ROUTER_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.1,
                max_tokens=200,
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
        logger.warning("intent_router parse failed call_id=%s raw=%r", state["call_id"], raw[:200])
        return _fallback(knn_intent, "knn_fallback_parse_error")

    return parsed
