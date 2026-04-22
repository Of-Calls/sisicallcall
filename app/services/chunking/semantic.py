import math
import re

from app.services.chunking.base import BaseChunkingService
from app.services.embedding.base import BaseEmbeddingService
from app.utils.logger import get_logger

logger = get_logger(__name__)

# TODO: PDF 길이·업종별 최적값 실험 필요 (현재 병원 FAQ ~4페이지 기준 99청크)
# SIMILARITY_THRESHOLD: 낮출수록 청크 수 증가. 0.70~0.80 범위에서 실험 권장
# MIN_CHUNK_LEN: 너무 낮으면 단문 청크 과다. 긴 PDF에서는 300~400 고려
# MAX_CHUNK_LEN: 표 포함 문서에서는 표가 중간에 잘릴 수 있음
SIMILARITY_THRESHOLD = 0.75
MIN_CHUNK_LEN = 100
MAX_CHUNK_LEN = 1000


class SemanticChunkingService(BaseChunkingService):
    """BGE-M3 임베딩 기반 시맨틱 청킹 — 의미 단위로 경계 감지."""

    def __init__(self, embedding_service: BaseEmbeddingService):
        self._embedding = embedding_service

    async def chunk(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        embeddings = await self._embedding.embed_batch(sentences)

        chunks: list[str] = []
        current: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            current_text = " ".join(current)

            boundary = sim < SIMILARITY_THRESHOLD or len(current_text) >= MAX_CHUNK_LEN
            if boundary:
                if len(current_text) >= MIN_CHUNK_LEN:
                    chunks.append(current_text)
                    current = [sentences[i]]
                else:
                    current.append(sentences[i])
            else:
                current.append(sentences[i])

        if current:
            last = " ".join(current)
            if chunks and len(last) < MIN_CHUNK_LEN:
                chunks[-1] = chunks[-1] + " " + last
            else:
                chunks.append(last)

        logger.debug(f"시맨틱 청킹 완료: {len(sentences)}문장 → {len(chunks)}청크")
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?。])\s+|\n{2,}", text)
        return [p.strip() for p in parts if p.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
