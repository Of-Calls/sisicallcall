"""STT Keyterm Cache — tenant ChromaDB llm_keywords 집계 + Redis 24h 캐싱.

통화 시작 시 tenant 컬렉션 전체 chunk metadata에서 llm_keywords를 빈도 집계해
Deepgram Nova-3 keyterm 부스팅 목록으로 변환. Redis TTL 24h 캐싱으로 반복 조회 제거.

Redis key: stt:keyterms:{tenant_id_no_hyphens}
ChromaDB: tenant_{tenant_id_no_hyphens}_docs 컬렉션
"""
import asyncio
import json
from collections import Counter

import redis.asyncio as aioredis

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_redis: aioredis.Redis = aioredis.from_url(settings.redis_url, decode_responses=True)

_KEYTERM_TTL_SEC = 86400   # 24h
_KEYTERM_TOP_N = 40        # Deepgram 500 토큰 제한 내 충분한 여유
_KEYTERM_MIN_LEN = 2       # 1자 조사/어미 제외


def _collection_name(tenant_id: str) -> str:
    return f"tenant_{tenant_id.replace('-', '')}_docs"


async def _fetch_all_metadatas(tenant_id: str) -> list[dict]:
    """ChromaDB 컬렉션 전체 chunk metadata 반환 (blocking → executor)."""
    def _query():
        import chromadb
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        try:
            col = client.get_collection(_collection_name(tenant_id))
        except Exception:
            return []
        result = col.get(include=["metadatas"])
        return result.get("metadatas") or []

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _query)


async def get_tenant_keyterms(tenant_id: str, top_n: int = _KEYTERM_TOP_N) -> list[str]:
    """tenant 전용 STT keyterm 목록 반환 (Redis 캐시 우선, miss 시 ChromaDB 집계).

    반환 리스트가 비어있으면 Deepgram keyterm 파라미터를 생략해도 무방하며,
    STT는 keyterm 없이 정상 동작한다.
    """
    redis_key = f"stt:keyterms:{tenant_id.replace('-', '')}"

    try:
        cached = await _redis.get(redis_key)
        if cached:
            logger.debug("keyterm_cache hit tenant=%s", tenant_id)
            return json.loads(cached)
    except Exception as e:
        logger.warning("keyterm_cache Redis get 실패 tenant=%s: %s", tenant_id, e)

    metadatas = await _fetch_all_metadatas(tenant_id)

    counter: Counter = Counter()
    for meta in metadatas:
        kws_str = (meta or {}).get("llm_keywords", "")
        for kw in kws_str.split(","):
            kw = kw.strip()
            if len(kw) >= _KEYTERM_MIN_LEN:
                counter[kw] += 1

    keyterms = [kw for kw, _ in counter.most_common(top_n)]

    logger.info("keyterm_cache miss→집계 tenant=%s chunks=%d keyterms=%d개", tenant_id, len(metadatas), len(keyterms))

    if keyterms:
        try:
            await _redis.setex(redis_key, _KEYTERM_TTL_SEC, json.dumps(keyterms, ensure_ascii=False))
        except Exception as e:
            logger.warning("keyterm_cache Redis setex 실패 tenant=%s: %s", tenant_id, e)

    return keyterms
