"""Clarify Branch — Intent Router 가 생성한 역질문(clarify_question) 을 그대로 응답으로 반환.

LLM 호출 없음, RAG 검색 없음. intent_router_llm_node 가 이미 만든 질문 텍스트를
TTS 단계로 흘려보내기만 한다. is_fallback=True 로 표기해 Semantic Cache 저장도 차단.

설계 의도:
    - 모호한 발화 ("아파", "글쎄요") 에 대해 도메인 맞춤 역질문으로 의도를 재확인
    - 별도 LLM 호출 없이 즉시 응답 → 5초 응답 제약 안에 충분히 마무리
    - 통화 다음 턴에서 사용자 답변을 last_assistant_text 맥락으로 해석 가능
"""
from app.agents.conversational.state import CallState
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CLARIFY = "조금 더 자세히 말씀해 주시겠어요?"


async def clarify_branch_node(state: CallState) -> dict:
    call_id = state.get("call_id", "unknown")
    clarify_text = (state.get("clarify_question") or "").strip() or _DEFAULT_CLARIFY

    logger.info(
        "clarify branch call_id=%s normalized_text=%r → question=%r",
        call_id, state.get("normalized_text", ""), clarify_text,
    )

    return {
        "response_text": clarify_text,
        "response_path": "clarify",
        "is_fallback": True,    # Semantic Cache 저장 차단 (질문은 캐싱 대상 아님)
        "is_timeout": False,
    }
