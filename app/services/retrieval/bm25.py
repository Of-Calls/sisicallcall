"""한국어 BM25 retrieval — Kiwi 형태소 분석 + rank_bm25.

Dense retrieval (Qwen3) 의 약점인 짧은 한국어 단어 ("위치", "단체") exact match 를 보강.
tenant 별 in-memory 인덱스 캐시. 첫 검색 시 ChromaDB 의 모든 청크로 build 후 재사용.
"""
from typing import Optional

from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi

from app.utils.logger import get_logger

logger = get_logger(__name__)

# 의미 있는 형태소 tag — 명사/대명사/용언 어간/외국어/숫자.
# 조사 (J*), 어미 (E*), 구두점 (S*), 부사 (MAG/MAJ), 감탄사 (IC) 등은 BM25 노이즈.
_KEEP_TAGS = {
    "NNG", "NNP", "NR", "NP",   # 일반/고유 명사, 수사, 대명사
    "VV", "VA",                  # 동사/형용사 어간
    "SL", "SH", "SN",            # 외국어, 한자, 숫자
}


class BM25Service:
    """tenant-별 in-memory BM25 인덱스 + Kiwi 토크나이저.

    Kiwi 인스턴스는 비싸므로 1회 생성 후 재사용.
    """

    def __init__(self):
        self._kiwi = Kiwi()
        # tenant_id → (BM25Okapi, [chunk_index, ...])
        self._cache: dict[str, tuple[BM25Okapi, list[int]]] = {}

    def tokenize(self, text: str) -> list[str]:
        """텍스트 → 의미 있는 형태소 list. 빈 입력 / 의미 형태소 없으면 빈 리스트."""
        if not text:
            return []
        tokens = self._kiwi.tokenize(text)
        return [t.form for t in tokens if t.tag in _KEEP_TAGS]

    def build(self, tenant_id: str, corpus: list[tuple[int, str]]) -> None:
        """corpus = [(chunk_index, text), ...] → tenant BM25 인덱스 cache.

        corpus 가 비면 캐시 evict.
        """
        if not corpus:
            self._cache.pop(tenant_id, None)
            logger.info("bm25 build skipped (empty corpus) tenant=%s", tenant_id[:8])
            return
        indices = [c[0] for c in corpus]
        tokenized = [self.tokenize(c[1]) for c in corpus]
        bm25 = BM25Okapi(tokenized)
        self._cache[tenant_id] = (bm25, indices)
        logger.info(
            "bm25 indexed tenant=%s chunks=%d avg_tokens=%.1f",
            tenant_id[:8], len(corpus),
            sum(len(t) for t in tokenized) / max(len(tokenized), 1),
        )

    def search(
        self, tenant_id: str, query: str, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """tenant BM25 인덱스에서 검색 → [(chunk_index, score), ...] (score 내림차순).

        score=0 인 항목은 제외 (관련 없음). top_k 가 부족하면 그만큼만 반환.
        """
        cached = self._cache.get(tenant_id)
        if cached is None:
            return []
        bm25, indices = cached
        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []
        scores = bm25.get_scores(q_tokens)
        ranked = [
            (ci, float(s)) for ci, s in zip(indices, scores) if s > 0
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def clear_cache(self, tenant_id: Optional[str] = None) -> None:
        """tenant_id 명시 시 그 tenant 만, None 이면 전체 evict."""
        if tenant_id is None:
            self._cache.clear()
        else:
            self._cache.pop(tenant_id, None)

    def has_cache(self, tenant_id: str) -> bool:
        return tenant_id in self._cache
