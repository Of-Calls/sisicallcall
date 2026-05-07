import asyncio
import time
import uuid

from app.services.cache.base import BaseCacheService, CacheHit
from app.utils.config import settings
from app.utils.logger import get_logger

# ChromaDB 컬렉션명: tenant_{tenant_id_without_hyphens}_cache
# RAG 컬렉션 (_docs) 와 분리 — 테넌트 격리 + 검색 영역 분리.

logger = get_logger(__name__)
_CHROMA_QUERY_PATH = "/api/v1/collections/{collection_id}/query"


class ChromaCacheService(BaseCacheService):
    def __init__(self):
        import chromadb

        self._client = chromadb.HttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        )

    def _collection_name(self, tenant_id: str) -> str:
        return f"tenant_{tenant_id.replace('-', '')}_cache"

    def _query_http_no_where_document(
        self,
        tenant_id: str,
        query_embedding: list[float],
        n_results: int,
        where: dict | None = None,
    ) -> dict:
        """col.query 대신 where_document 키를 생략한 REST 호출."""
        import httpx

        col = self._client.get_or_create_collection(self._collection_name(tenant_id))
        collection_id = str(col.id)
        url = (
            f"http://{settings.chroma_host}:{settings.chroma_port}"
            f"{_CHROMA_QUERY_PATH.format(collection_id=collection_id)}"
        )
        body: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["metadatas", "distances"],
        }
        if where is not None:
            body["where"] = where
        with httpx.Client(timeout=120.0) as http:
            resp = http.post(url, json=body)
            resp.raise_for_status()
            return resp.json()

    async def lookup(
        self, tenant_id: str, query_embedding: list[float]
    ) -> CacheHit | None:
        loop = asyncio.get_event_loop()

        def _query():
            now = time.time()
            result = self._query_http_no_where_document(
                tenant_id=tenant_id,
                query_embedding=query_embedding,
                n_results=1,
                where={"expires_at": {"$gt": now}},
            )
            dists = (result.get("distances") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            if not dists or not metas:
                return None
            distance = dists[0]
            if distance > settings.cache_distance_threshold:
                return None
            meta = metas[0] or {}
            if meta.get("expires_at", 0) <= now:
                return None
            return CacheHit(
                response_text=meta.get("response_text", ""),
                distance=float(distance),
                cached_at=float(meta.get("created_at", 0)),
            )

        return await loop.run_in_executor(None, _query)

    async def save(
        self,
        tenant_id: str,
        query_text: str,
        query_embedding: list[float],
        response_text: str,
    ) -> None:
        loop = asyncio.get_event_loop()
        now = time.time()
        expires_at = now + settings.cache_ttl_seconds
        entry_id = str(uuid.uuid4())

        def _save():
            col = self._client.get_or_create_collection(
                self._collection_name(tenant_id)
            )
            col.add(
                ids=[entry_id],
                embeddings=[query_embedding],
                documents=[query_text],
                metadatas=[
                    {
                        "query_text": query_text,
                        "response_text": response_text,
                        "created_at": now,
                        "expires_at": expires_at,
                    }
                ],
            )

        await loop.run_in_executor(None, _save)
        logger.info("cache save tenant=%s id=%s", tenant_id, entry_id)

    async def clear(self, tenant_id: str) -> None:
        loop = asyncio.get_event_loop()

        def _clear():
            try:
                self._client.delete_collection(self._collection_name(tenant_id))
            except Exception as exc:
                logger.warning("cache clear failed tenant=%s err=%s", tenant_id, exc)

        await loop.run_in_executor(None, _clear)
        logger.info("cache clear tenant=%s", tenant_id)
