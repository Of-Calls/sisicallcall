from datetime import datetime, time, timedelta, timezone
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

# TODO(tenant.settings 이관): 업종 확장 시 tenant별 timezone 필드로 전환
# KST는 DST 없어 영원히 +09:00이므로 IANA zoneinfo(tzdata 의존) 대신 고정 offset 사용
_KST = timezone(timedelta(hours=9))


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
        now = now or datetime.now(_KST)
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

        current = now.time()
        # 야간업종 대응: start > end이면 자정을 건너뛰는 운영시간 (예: "22:00-02:00")
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end

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

    # RFC 001 v0.2 §6.5 — run_turn 진입 시 pre-load 되어 CallState["stall_messages"] 에 주입됨
    _DEFAULT_STALL_MESSAGES = {"general": "잠시만요, 확인해 드리겠습니다."}

    async def get_stall_messages(self, tenant_id: str) -> dict:
        """대기 멘트 문구 조회. 키 부재/에러 시 하드코딩 기본값 반환.

        반환 dict 는 반드시 'general' 키를 포함한다 (브랜치별 fallback 용도).
        """
        key = self._tenant_key(tenant_id, "stall_messages")
        try:
            value = await self._redis.hgetall(key)
        except Exception as e:
            logger.error("redis stall_messages lookup failed tenant=%s: %s", tenant_id, e)
            return dict(self._DEFAULT_STALL_MESSAGES)

        if not value or "general" not in value:
            # general 누락 시 기본값으로 병합 (브랜치별 문구만 있고 general 없는 엣지 케이스 방어)
            merged = dict(self._DEFAULT_STALL_MESSAGES)
            merged.update(value or {})
            return merged
        return value
