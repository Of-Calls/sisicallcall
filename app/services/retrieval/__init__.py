"""Retrieval 모듈 — Dense (Qwen3 + ChromaDB) + BM25 (Kiwi) hybrid.

Path A (2026-05-06) — 한국어 짧은 단어 query 의 lexical exact match 보강.
Dense retrieval 의 약점 (예: "위치", "단체", "전화번호" 같은 짧은 명사 query 가
긴 자연어 청크와 cosine 거리 멀어지는 패턴) 을 BM25 로 메움.

singleton 운영:
- `get_hybrid_retriever()` 가 module-level singleton 반환 — main.py 의 prewarm
  결과 (BM25 인덱스 캐시) 를 faq_branch 와 공유.
"""
from __future__ import annotations

from typing import Optional

from app.services.embedding.base import BaseEmbeddingService
from app.services.rag.chroma import ChromaRAGService
from app.services.retrieval.bm25 import BM25Service
from app.services.retrieval.hybrid import HybridRetriever
from app.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["BM25Service", "HybridRetriever", "get_hybrid_retriever", "prewarm_all_tenants"]


_hybrid: HybridRetriever | None = None


def get_hybrid_retriever(
    rag: Optional[ChromaRAGService] = None,
    embedder: Optional[BaseEmbeddingService] = None,
) -> HybridRetriever:
    """module-level singleton — 모든 호출자가 같은 BM25 캐시 공유."""
    global _hybrid
    if _hybrid is None:
        _hybrid = HybridRetriever(
            rag=rag or ChromaRAGService(),
            embedder=embedder,
        )
    return _hybrid


async def prewarm_all_tenants() -> None:
    """active tenant 의 BM25 인덱스 미리 build. 실패해도 다음 tenant 진행 (graceful)."""
    import asyncpg
    from app.utils.config import settings

    hybrid = get_hybrid_retriever()

    conn = await asyncpg.connect(settings.database_url)
    try:
        rows = await conn.fetch("SELECT id, name FROM tenants")
    finally:
        await conn.close()

    for r in rows:
        tid = str(r["id"])
        try:
            await hybrid._ensure_bm25(tid)
        except Exception as e:
            logger.warning("bm25 prewarm failed tenant=%s name=%s err=%s", tid[:8], r["name"], e)
