"""Retrieval 모듈 — Dense (Qwen3 + ChromaDB) + BM25 (Kiwi) hybrid.

Path A (2026-05-06) — 한국어 짧은 단어 query 의 lexical exact match 보강.
Dense retrieval 의 약점 (예: "위치", "단체", "전화번호" 같은 짧은 명사 query 가
긴 자연어 청크와 cosine 거리 멀어지는 패턴) 을 BM25 로 메움.
"""
from app.services.retrieval.bm25 import BM25Service
from app.services.retrieval.hybrid import HybridRetriever

__all__ = ["BM25Service", "HybridRetriever"]
