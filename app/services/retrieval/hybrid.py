"""Dense (Qwen3 + ChromaDB) + BM25 (Kiwi) hybrid retrieval — RRF 결합.

Reciprocal Rank Fusion (k=60) — 점수 정규화 불필요, rank 만 사용.
    score(c) = 1/(k + rank_dense(c)) + 1/(k + rank_bm25(c))

dense 와 bm25 가 둘 다 잡으면 score 합산 → 상위. 둘 중 하나만 잡아도 상위 가능.
"""
from typing import Optional

from app.services.embedding.base import BaseEmbeddingService
from app.services.rag.chroma import ChromaRAGService
from app.services.retrieval.bm25 import BM25Service
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_RRF_K = 60


class HybridRetriever:
    """Dense + BM25 결합 검색기. tenant 별 BM25 인덱스 lazy build (첫 검색 시)."""

    def __init__(
        self,
        rag: ChromaRAGService,
        embedder: Optional[BaseEmbeddingService] = None,
        bm25: Optional[BM25Service] = None,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        """embedder 는 search() 가 query 임베딩 자동 호출 시에만 필요.
        search_with_embedding() 은 외부에서 주입하므로 embedder=None OK.
        """
        self._embedder = embedder
        self._rag = rag
        self._bm25 = bm25 or BM25Service()
        self._rrf_k = rrf_k

    async def _ensure_bm25(self, tenant_id: str) -> None:
        """tenant BM25 인덱스 lazy build. 이미 build 됐으면 skip."""
        if self._bm25.has_cache(tenant_id):
            return
        all_chunks = await self._rag.list_chunks(tenant_id)
        corpus = [
            (c["chunk_index"], c["document"])
            for c in all_chunks
            if c.get("chunk_index") is not None and c.get("document")
        ]
        self._bm25.build(tenant_id, corpus)

    def invalidate(self, tenant_id: str) -> None:
        """tenant reseed 후 호출 — BM25 인덱스 evict (다음 검색 시 rebuild)."""
        self._bm25.clear_cache(tenant_id)

    async def search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 5,
        top_dense: int = 10,
        top_bm25: int = 10,
    ) -> list[dict]:
        """Hybrid 검색 — dense + BM25 RRF 결합 후 top_k 반환.

        embedder 가 None 이면 호출 불가 — search_with_embedding 사용.
        """
        if self._embedder is None:
            raise RuntimeError("HybridRetriever.search() requires embedder. Use search_with_embedding() instead.")
        embedding = await self._embedder.embed_query(query)
        return await self.search_with_embedding(
            query, embedding, tenant_id,
            top_k=top_k, top_dense=top_dense, top_bm25=top_bm25,
        )

    async def search_with_embedding(
        self,
        query: str,
        query_embedding: list[float],
        tenant_id: str,
        top_k: int = 5,
        top_dense: int = 10,
        top_bm25: int = 10,
    ) -> list[dict]:
        """external embedding 주입 검색 — cache 와 dense 측에 같은 embedding 재사용.

        반환 항목: {chunk_index, document, metadata, distance, bm25_score, rrf_score,
        dense_rank, bm25_rank}. 어느 한 retriever 만 잡으면 그 쪽만 score 가짐.
        """
        dense_results = await self._rag.search_with_meta(
            query_embedding, tenant_id, top_k=top_dense
        )

        await self._ensure_bm25(tenant_id)
        bm25_results = self._bm25.search(tenant_id, query, top_k=top_bm25)

        rrf: dict[int, dict] = {}

        for rank, r in enumerate(dense_results):
            ci = (r.get("metadata") or {}).get("chunk_index")
            if ci is None:
                continue
            rrf[ci] = {
                "chunk_index": ci,
                "document": r.get("document", ""),
                "metadata": r.get("metadata") or {},
                "distance": r.get("distance"),
                "bm25_score": 0.0,
                "rrf_score": 1.0 / (self._rrf_k + rank + 1),
                "dense_rank": rank + 1,
                "bm25_rank": None,
            }

        bm25_only_indices: list[int] = []
        for rank, (ci, score) in enumerate(bm25_results):
            entry = rrf.get(ci)
            if entry is None:
                rrf[ci] = {
                    "chunk_index": ci,
                    "document": "",
                    "metadata": {},
                    "distance": None,
                    "bm25_score": score,
                    "rrf_score": 1.0 / (self._rrf_k + rank + 1),
                    "dense_rank": None,
                    "bm25_rank": rank + 1,
                }
                bm25_only_indices.append(ci)
            else:
                entry["bm25_score"] = score
                entry["bm25_rank"] = rank + 1
                entry["rrf_score"] += 1.0 / (self._rrf_k + rank + 1)

        # BM25 만 잡은 청크는 ChromaDB 에서 document/metadata 보충 필요 (humanize 단계에 사용).
        if bm25_only_indices:
            await self._enrich_bm25_only(tenant_id, rrf, bm25_only_indices)

        ranked = sorted(rrf.values(), key=lambda x: x["rrf_score"], reverse=True)
        return ranked[:top_k]

    async def _enrich_bm25_only(
        self, tenant_id: str, rrf: dict[int, dict], indices: list[int]
    ) -> None:
        """BM25 만 hit 한 청크의 document/metadata 를 ChromaDB get 으로 보충."""
        all_chunks = await self._rag.list_chunks(tenant_id)
        idx_map = {c.get("chunk_index"): c for c in all_chunks}
        for ci in indices:
            chunk = idx_map.get(ci)
            if chunk is None:
                continue
            entry = rrf[ci]
            entry["document"] = chunk.get("document", "")
            entry["metadata"] = chunk.get("metadata") or {}
