"""RAG probe — cache miss 직후, intent_router 진입 전.

ChromaDB top_k=3 검색을 1회 가볍게 실행해 router LLM 에게 RAG 강/약 신호를
state["rag_probe"] 로 노출한다. router 가 이 신호를 보고 더 정확하게 intent
분류 가능 (특히 모호 발화 + 강 RAG 신호, 명확 발화 + RAG miss 케이스).

신호 (state["rag_probe"]):
    {
        "top_distance": float,           # cosine distance, 0~2 (낮을수록 유사)
        "matched_keywords": list[str],   # query 와 chunk.llm_keywords substring 매칭
        "top_topic": str,                # chunk.llm_topic (clarify 유도질문용)
        "top_title": str,                # chunk.llm_title (디버그)
        "top_chunk_id": str,             # 디버그/추적용
    }
오류·미보유 시 state["rag_probe"] = None.

검색 자체는 faq_branch_node 가 다시 수행 (top_k=8, hybrid score, distance filter).
즉 RAG 검색이 cache miss turn 마다 2회 발생 — 신호용/답변용. 후속 최적화 후보:
  - faq_branch 가 probe 결과 (top_k=3) 재사용해 추가 query 생략
"""
import asyncio

from app.agents.conversational.state import CallState
from app.services.rag.base import BaseRAGService
from app.services.rag.chroma import ChromaRAGService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_rag: BaseRAGService = ChromaRAGService()

PROBE_TOP_K = 3
PROBE_TIMEOUT_SEC = 1.0  # 5초 hardcut 안전 마진. ChromaDB local query ~30~80ms 추정.


async def rag_probe_node(state: CallState) -> dict:
    call_id = state["call_id"]
    query_embedding = state.get("query_embedding") or []
    if not query_embedding:
        return {"rag_probe": None}

    try:
        results = await asyncio.wait_for(
            _rag.search_with_meta(
                query_embedding=query_embedding,
                tenant_id=state["tenant_id"],
                top_k=PROBE_TOP_K,
            ),
            timeout=PROBE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning("rag_probe timeout call_id=%s", call_id)
        return {"rag_probe": None}
    except Exception as e:
        logger.error("rag_probe error call_id=%s: %s", call_id, e)
        return {"rag_probe": None}

    if not results:
        return {"rag_probe": None}

    top = results[0]
    meta = top.get("metadata") or {}
    query_text = state.get("normalized_text") or ""
    keywords = [k.strip() for k in (meta.get("llm_keywords") or "").split(",") if k.strip()]
    matched = [kw for kw in keywords if kw in query_text]

    probe = {
        "top_distance": top.get("distance"),
        "matched_keywords": matched,
        "top_topic": meta.get("llm_topic", ""),
        "top_title": meta.get("llm_title", ""),
        "top_chunk_id": top.get("id", ""),
    }
    logger.info(
        "rag_probe call_id=%s distance=%.3f matched=%s topic=%r",
        call_id,
        probe["top_distance"] if probe["top_distance"] is not None else -1.0,
        matched, probe["top_topic"],
    )
    return {"rag_probe": probe}
