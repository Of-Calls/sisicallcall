import asyncio
import hashlib
import json
import math
import time

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Redis 키 (CLAUDE.md 규칙: 콜론 구분, 소문자, UUID 하이픈 제거)
#   cache:{tenant_no_hyphens}:{text_hash}  → Hash {embedding, response_text, cache_source, created_at}
#   cache_idx:{tenant_no_hyphens}          → Set of text_hashes (의미 유사도 검색용 인덱스)
#
# 저장 금지 조건은 cache_store_node 에서 검사 (CLAUDE.md):
# - 타임아웃 폴백 / Escalation 응답 / Reviewer revise 응답 / cache 경로 응답

SIMILARITY_THRESHOLD = 0.95     # cosine similarity 임계값 (보수적 — false hit 방지)
TTL_SECONDS = 60 * 60 * 24      # 24시간


def _tenant_key(tenant_id: str) -> str:
    return tenant_id.replace("-", "").lower()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCacheService:
    """2단 매칭 캐시: 텍스트 정확 일치 → 임베딩 cosine similarity."""

    def __init__(self):
        import redis.asyncio as aioredis
        self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    async def lookup(
        self, text: str, tenant_id: str, embedding: list[float]
    ) -> dict | None:
        tenant_key = _tenant_key(tenant_id)

        # 1차: text_hash 정확 일치 (~10ms)
        exact_key = f"cache:{tenant_key}:{_text_hash(text)}"
        raw = await self._redis.hgetall(exact_key)
        if raw:
            try:
                return {
                    "embedding": json.loads(raw["embedding"]),
                    "response_text": raw["response_text"],
                }
            except (KeyError, json.JSONDecodeError):
                logger.warning("cache exact entry corrupt key=%s", exact_key)

        # 2차: tenant 인덱스 순회 + cosine similarity
        idx_key = f"cache_idx:{tenant_key}"
        hashes = await self._redis.smembers(idx_key)
        if not hashes:
            return None

        # asyncio.gather 로 동시 조회 (redis-py async pipeline 호환성 이슈 회피)
        hash_list = list(hashes)
        entries = await asyncio.gather(
            *(self._redis.hgetall(f"cache:{tenant_key}:{h}") for h in hash_list)
        )

        best_score = 0.0
        best_response = None
        best_embedding = None
        stale_hashes = []
        for h, entry in zip(hash_list, entries):
            if not entry:
                # 인덱스에는 있지만 항목 없음 (TTL 만료) → 인덱스 정리
                stale_hashes.append(h)
                continue
            try:
                cached_embedding = json.loads(entry["embedding"])
            except (KeyError, json.JSONDecodeError):
                continue
            score = _cosine_similarity(embedding, cached_embedding)
            if score > best_score:
                best_score = score
                best_response = entry["response_text"]
                best_embedding = cached_embedding

        if stale_hashes:
            await self._redis.srem(idx_key, *stale_hashes)

        if best_score >= SIMILARITY_THRESHOLD:
            logger.debug("cache semantic hit similarity=%.4f", best_score)
            return {
                "embedding": best_embedding,
                "response_text": best_response,
            }
        return None

    async def store(
        self,
        text: str,
        tenant_id: str,
        embedding: list[float],
        response_text: str,
        cache_source: str,
    ) -> None:
        tenant_key = _tenant_key(tenant_id)
        text_hash = _text_hash(text)
        entry_key = f"cache:{tenant_key}:{text_hash}"
        idx_key = f"cache_idx:{tenant_key}"

        await self._redis.hset(
            entry_key,
            mapping={
                "embedding": json.dumps(embedding),
                "response_text": response_text,
                "cache_source": cache_source,
                "created_at": str(int(time.time())),
            },
        )
        await self._redis.expire(entry_key, TTL_SECONDS)
        await self._redis.sadd(idx_key, text_hash)
