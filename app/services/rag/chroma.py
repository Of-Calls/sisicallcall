from app.services.rag.base import BaseRAGService
from app.utils.config import settings
from app.utils.logger import get_logger

# ChromaDB 컬렉션명: tenant_{tenant_id_without_hyphens}_docs

logger = get_logger(__name__)

# chromadb Python HttpClient 가 col.query 시 JSON 에 "where": {} 를 실어 보내고,
# Chroma 서버(0.5.x)는 빈 where 를 InvalidArgumentError 로 거절함 → 필터 없는 검색은
# where 키 자체를 생략한 REST 호출로 처리.

_CHROMA_QUERY_PATH = "/api/v1/collections/{collection_id}/query"


class ChromaRAGService(BaseRAGService):
    def __init__(self):
        import chromadb
        self._client = chromadb.HttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        )

    def _collection_name(self, tenant_id: str) -> str:
        return f"tenant_{tenant_id.replace('-', '')}_docs"

    def _query_http_no_where(
        self,
        tenant_id: str,
        query_embedding: list[float],
        n_results: int,
        *,
        include: list[str] | None,
    ) -> dict:
        """col.query 대신 JSON 본문에 where 를 넣지 않고 POST (유사도만 top-k)."""
        import httpx

        col = self._client.get_or_create_collection(self._collection_name(tenant_id))
        collection_id = str(col.id)
        path = _CHROMA_QUERY_PATH.format(collection_id=collection_id)
        base = f"http://{settings.chroma_host}:{settings.chroma_port}"
        url = f"{base}{path}"
        body: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if include is not None:
            body["include"] = include
        with httpx.Client(timeout=120.0) as http:
            r = http.post(url, json=body)
            r.raise_for_status()
            return r.json()

    async def search(
        self, query_embedding: list[float], tenant_id: str, top_k: int = 3
    ) -> list[str]:
        """벡터 유사도 검색 — FAQ 브랜치 전용."""
        import asyncio

        loop = asyncio.get_event_loop()

        def _query():
            result = self._query_http_no_where(
                tenant_id,
                query_embedding,
                top_k,
                include=["metadatas", "documents", "distances"],
            )
            return result["documents"][0] if result.get("documents") else []

        return await loop.run_in_executor(None, _query)

    async def search_with_meta(
        self, query_embedding: list[float], tenant_id: str, top_k: int = 3
    ) -> list[dict]:
        """벡터 검색 + id/distance/metadata 동봉 반환 — 진단/로깅용."""
        import asyncio

        loop = asyncio.get_event_loop()

        def _query():
            result = self._query_http_no_where(
                tenant_id,
                query_embedding,
                top_k,
                include=["documents", "metadatas", "distances"],
            )
            docs_outer = result.get("documents") or []
            if not docs_outer:
                return []
            ids = (result.get("ids") or [[]])[0]
            docs = docs_outer[0] or []
            metas = (result.get("metadatas") or [[]])[0]
            dists = (result.get("distances") or [[]])[0]
            out: list[dict] = []
            for i, doc in enumerate(docs):
                out.append({
                    "id": ids[i] if i < len(ids) else "",
                    "document": doc,
                    "distance": dists[i] if i < len(dists) else None,
                    "metadata": metas[i] if i < len(metas) else {},
                })
            return out

        return await loop.run_in_executor(None, _query)

    async def upsert(
        self,
        doc_id: str,
        content: str,
        embedding: list[float],
        tenant_id: str,
        metadata: dict,
    ) -> None:
        """RAG 문서 저장 (소프트 삭제 시 ChromaDB 벡터 동시 삭제 필수)."""
        import asyncio

        loop = asyncio.get_event_loop()

        def _upsert():
            col = self._client.get_or_create_collection(self._collection_name(tenant_id))
            col.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata],
            )

        await loop.run_in_executor(None, _upsert)
        logger.info("chroma upsert doc_id=%s tenant=%s", doc_id, tenant_id)

    async def delete(self, doc_id: str, tenant_id: str) -> None:
        """소프트 삭제 시 ChromaDB 벡터 동시 삭제 (db_schema.md 규칙)."""
        import asyncio

        loop = asyncio.get_event_loop()

        def _delete():
            col = self._client.get_or_create_collection(self._collection_name(tenant_id))
            col.delete(ids=[doc_id])

        await loop.run_in_executor(None, _delete)
        logger.info("chroma delete doc_id=%s tenant=%s", doc_id, tenant_id)

    async def delete_by_document(self, document_id: str, tenant_id: str) -> None:
        """document_id에 속한 모든 청크 삭제 — 문서 교체/삭제 시 사용."""
        import asyncio

        loop = asyncio.get_event_loop()

        def _delete():
            col = self._client.get_or_create_collection(self._collection_name(tenant_id))
            col.delete(where={"document_id": {"$eq": document_id}})

        await loop.run_in_executor(None, _delete)
        logger.info("chroma delete_by_document document_id=%s tenant=%s", document_id, tenant_id)
