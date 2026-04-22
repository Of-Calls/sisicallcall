from app.agents.conversational.state import CallState
from app.services.cache.semantic_cache import SemanticCacheService
from app.services.embedding.base import BaseEmbeddingService
from app.services.embedding.local import BGEM3LocalEmbeddingService
from app.utils.logger import get_logger

# 임베딩은 Cache 노드에서 1회만 생성하고 KNN/FAQ 노드가 재사용 (langgraph_spec §4.4)
# TODO(BGE-M3 이관): 희영 R-02 완료 후 로컬/API 구현체 확정 교체
_embedding_service: BaseEmbeddingService = BGEM3LocalEmbeddingService()
_cache_service = SemanticCacheService()

logger = get_logger(__name__)


async def cache_node(state: CallState) -> dict:
    try:
        query_embedding = await _embedding_service.embed(state["normalized_text"])
    except Exception as e:
        logger.error("embedding failed call_id=%s: %s", state["call_id"], e)
        return {"query_embedding": [], "cache_hit": False}

    try:
        result = await _cache_service.lookup(
            text=state["normalized_text"],
            tenant_id=state["tenant_id"],
        )
    except Exception as e:
        logger.error("cache lookup failed call_id=%s: %s", state["call_id"], e)
        return {"query_embedding": query_embedding, "cache_hit": False}

    if result:
        return {
            "query_embedding": query_embedding,
            "cache_hit": True,
            "response_text": result["response_text"],
            "response_path": "cache",
        }
    return {"query_embedding": query_embedding, "cache_hit": False}
