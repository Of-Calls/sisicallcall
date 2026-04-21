from datetime import datetime, time
from typing import Optional

from app.utils.config import settings
from app.utils.logger import get_logger

# Redis 세션/테넌트 공용 서비스.
# 키 구조 (feature_spec.md §6.5):
#   tenant:{tenant_id}:business_hours   — Hash, 필드 mon/tue/.../sun, 값 "HH:MM-HH:MM" | "closed"
#   tenant:{tenant_id}:agent_availability — Hash, 필드 "available" = 현재 가용 상담원 수
# tenant_id UUID 하이픈 제거 규칙은 CLAUDE.md 네이밍 규칙 준수.

logger = get_logger(__name__)

_WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class RedisSessionService:
    def __init__(self):
        import redis.asyncio as aioredis
        self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    def _tenant_key(self, tenant_id: str, suffix: str) -> str:
        return f"tenant:{tenant_id.replace('-', '')}:{suffix}"

    async def is_within_business_hours(
        self, tenant_id: str, now: Optional[datetime] = None
    ) -> bool:
        """현재 시각이 tenant 운영시간 내인지 판정. 데이터 부재/에러 시 False."""
        now = now or datetime.now()
        weekday = _WEEKDAYS[now.weekday()]
        key = self._tenant_key(tenant_id, "business_hours")
        try:
            value = await self._redis.hget(key, weekday)
        except Exception as e:
            logger.error("redis business_hours lookup failed tenant=%s: %s", tenant_id, e)
            return False

        if not value or value == "closed":
            return False

        try:
            start_str, end_str = value.split("-")
            start = time.fromisoformat(start_str.strip())
            end = time.fromisoformat(end_str.strip())
        except Exception as e:
            logger.error("business_hours parse error tenant=%s value=%s: %s", tenant_id, value, e)
            return False

        return start <= now.time() <= end

    async def get_available_agent_count(self, tenant_id: str) -> int:
        """상담원 가용 수 조회. 키 부재/에러 시 0 반환."""
        key = self._tenant_key(tenant_id, "agent_availability")
        try:
            value = await self._redis.hget(key, "available")
        except Exception as e:
            logger.error("redis agent_availability lookup failed tenant=%s: %s", tenant_id, e)
            return 0

        try:
            return int(value) if value else 0
        except (ValueError, TypeError):
            return 0
