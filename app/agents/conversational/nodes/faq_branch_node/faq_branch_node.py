from app.agents.conversational.state import CallState
from app.agents.conversational.utils.stall import FALLBACK_MESSAGE, _run_with_stall
from app.services.llm.base import BaseLLMService
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.rag.base import BaseRAGService
from app.services.rag.chroma import ChromaRAGService
from app.utils.logger import get_logger

# FAQ 브랜치 — RAG + GPT-4o-mini 3초 하드컷 (RFC 001 v0.2 §6.8)
# stall trigger 1초, TTSOutputChannel 경유

logger = get_logger(__name__)

_rag: BaseRAGService = ChromaRAGService()
_llm: BaseLLMService = GPT4OMiniService()

FAQ_LLM_TIMEOUT_SEC = 3.0
RAG_TOP_K = 3

# TODO(agents.md 이관): 담당자 배정 후 프롬프트를 agents.md 로 이관
FAQ_SYSTEM_PROMPT = """당신은 고객센터 FAQ 응답 AI입니다.
반드시 아래 규칙을 지키세요.
1) 제공된 '참고 자료' 범위 안에서만 답변합니다.
2) 참고 자료에 답이 없으면 정확히 다음 문장으로 답합니다: "확인이 어려워 담당자에게 연결해 드리겠습니다."
3) 한국어 존댓말로 2~3문장 이내로 간결하게 답합니다.
4) 참고 자료에 없는 정보를 추측하거나 생성하지 마세요."""


def _compose_user_message(normalized_text: str, rag_results: list[str]) -> str:
    if rag_results:
        joined = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(rag_results))
    else:
        joined = "(참고 자료 없음)"
    return f"참고 자료:\n{joined}\n\n고객 질문: {normalized_text}"


def _pick_stall_msg(state: CallState) -> str:
    """CallState 에서 FAQ 대기 멘트 선택. 없으면 general → 하드코딩 순으로 fallback."""
    msgs = state.get("stall_messages") or {}
    return msgs.get("faq") or msgs.get("general") or "잠시만요, 확인해 드리겠습니다."


async def faq_branch_node(state: CallState) -> dict:
    call_id = state["call_id"]
    query_embedding = state.get("query_embedding") or []

    # RAG 검색 — 임베딩 없으면 스킵 (cache_node 에서 실패한 경우)
    rag_results: list[str] = []
    if query_embedding:
        try:
            rag_results = await _rag.search(
                query_embedding=query_embedding,
                tenant_id=state["tenant_id"],
                top_k=RAG_TOP_K,
            )
        except Exception as e:
            logger.error("rag search failed call_id=%s: %s", call_id, e)

    user_message = _compose_user_message(state["normalized_text"], rag_results)

    # LLM 호출 + stall trigger race — RFC 001 v0.2 §6.5
    response_text, is_timeout = await _run_with_stall(
        coro=_llm.generate(
            system_prompt=FAQ_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.1,
            max_tokens=300,
        ),
        call_id=call_id,
        stall_msg=_pick_stall_msg(state),
        stall_audio_field="faq",
        delay=state.get("stall_delay_sec", 1.0),
        hardcut_sec=FAQ_LLM_TIMEOUT_SEC,
        rag_results=rag_results,
        fallback_text=FALLBACK_MESSAGE,
    )

    # LLM 이 공백만 반환한 엣지 케이스도 FALLBACK 으로 보정
    if not response_text:
        response_text = FALLBACK_MESSAGE

    return {
        "rag_results": rag_results,
        "response_text": response_text,
        "response_path": "faq",
        "is_timeout": is_timeout,
    }
