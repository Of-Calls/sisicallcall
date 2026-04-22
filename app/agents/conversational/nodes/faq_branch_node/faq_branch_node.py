import asyncio

from app.agents.conversational.state import CallState
from app.services.llm.base import BaseLLMService
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.rag.base import BaseRAGService
from app.services.rag.chroma import ChromaRAGService
from app.utils.logger import get_logger

# FAQ 브랜치 — RAG + GPT-4o-mini 2초 하드컷 (langgraph_spec.md §6.2, feature_spec.md §4.5)
# 타임아웃 시 RAG top-1 청크 원본을 그대로 전달 (자연어 재구성 생략)

logger = get_logger(__name__)

_rag: BaseRAGService = ChromaRAGService()
_llm: BaseLLMService = GPT4OMiniService()

FAQ_LLM_TIMEOUT_SEC = 3.0
RAG_TOP_K = 3

FALLBACK_MESSAGE = "확인이 어려워 담당자에게 연결해 드리겠습니다."

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


async def faq_branch_node(state: CallState) -> dict:
    call_id = state["call_id"]
    query_embedding = state.get("query_embedding") or []

    # RAG 검색 — 임베딩 없으면 스킵 (cache_node에서 실패한 경우)
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

    try:
        response_text = await asyncio.wait_for(
            _llm.generate(
                system_prompt=FAQ_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.1,
                max_tokens=300,
            ),
            timeout=FAQ_LLM_TIMEOUT_SEC,
        )
        return {
            "rag_results": rag_results,
            "response_text": response_text.strip() or FALLBACK_MESSAGE,
            "response_path": "faq",
            "is_timeout": False,
        }
    except asyncio.TimeoutError:
        logger.warning("faq branch timeout call_id=%s", call_id)
        fallback = rag_results[0] if rag_results else FALLBACK_MESSAGE
        return {
            "rag_results": rag_results,
            "response_text": fallback,
            "response_path": "faq",
            "is_timeout": True,
        }
    except Exception as e:
        logger.error("faq branch error call_id=%s: %s", call_id, e)
        return {
            "rag_results": rag_results,
            "response_text": FALLBACK_MESSAGE,
            "response_path": "faq",
            "is_timeout": False,
        }
