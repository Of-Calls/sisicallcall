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
        """현재 시각이 tenant 운영시간 내인지 판정.

        데이터 부재/에러/closed 시 default True (영업중 가정). 운영 환경에선 Redis seed 가
        채워져 있으니 정상 시각 판정. 개발/테스트 편의용 default. 영업시간 외를
        시뮬레이션하려면 escalation_branch_node 의 FORCE_OFFHOURS 환경변수 사용.
        """
        now = now or datetime.now(_KST)
        weekday = _WEEKDAYS[now.weekday()]
        key = self._tenant_key(tenant_id, "business_hours")
        try:
            value = await self._redis.hget(key, weekday)
        except Exception as e:
            logger.error("redis business_hours lookup failed tenant=%s: %s", tenant_id, e)
            return True

        if not value or value == "closed":
            return True

        try:
            start_str, end_str = value.split("-")
            start = time.fromisoformat(start_str.strip())
            end = time.fromisoformat(end_str.strip())
        except Exception as e:
            logger.error("business_hours parse error tenant=%s value=%s: %s", tenant_id, value, e)
            return True

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

    async def set_rag_categories(self, tenant_id: str, categories: list[str]) -> None:
        """tenant 가용 RAG 카테고리 자연어 list 저장.

        pdf_processor 가 PDF 처리 후 LLM 정제 결과를 write. JSON array str 로 저장.
        TTL 없음 — PDF 재처리 시 덮어씀, tenant 삭제 시 별도 cleanup.
        """
        import json
        key = self._tenant_key(tenant_id, "rag_categories")
        try:
            await self._redis.set(key, json.dumps(categories, ensure_ascii=False))
            logger.info(
                "rag_categories saved tenant=%s count=%d",
                tenant_id, len(categories),
            )
        except Exception as e:
            logger.error("redis rag_categories save failed tenant=%s: %s", tenant_id, e)

    async def get_rag_categories(self, tenant_id: str) -> list[str]:
        """tenant 가용 RAG 카테고리 자연어 list 반환.

        Redis 키: tenant:{tenant_id_no_hyphens}:rag_categories — JSON array str.
        pdf_processor 가 PDF 처리 후 LLM 정제 결과를 write. 부재/에러 시 빈 list.
        faq_branch_node 가 rag_miss_count >= 2 일 때 안내 멘트 생성에 사용.
        """
        import json
        key = self._tenant_key(tenant_id, "rag_categories")
        try:
            value = await self._redis.get(key)
        except Exception as e:
            logger.error("redis rag_categories lookup failed tenant=%s: %s", tenant_id, e)
            return []

        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(c) for c in parsed if c]
            return []
        except Exception as e:
            logger.error("rag_categories parse error tenant=%s value=%s: %s", tenant_id, value, e)
            return []

