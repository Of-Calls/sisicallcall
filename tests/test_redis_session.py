"""RedisSessionService 유닛 테스트.

is_within_business_hours / get_available_agent_count / _tenant_key 검증.
외부 Redis 의존 없이 AsyncMock으로 _redis를 교체한다.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from app.services.session.redis_session import RedisSessionService

_KST = timezone(timedelta(hours=9))

# 2026-04-20(Mon), 2026-04-21(Tue), ... 2026-04-26(Sun)
def _kst(year, month, day, hour, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=_KST)


@pytest.fixture
def service():
    s = RedisSessionService()
    s._redis = AsyncMock()
    return s


# ---------------------------------------------------------------------------
# _tenant_key
# ---------------------------------------------------------------------------

def test_tenant_key_strips_hyphens(service):
    key = service._tenant_key("a1b2c3d4-5678-90ab-cdef-1234567890ab", "business_hours")
    assert key == "tenant:a1b2c3d4567890abcdef1234567890ab:business_hours"


def test_tenant_key_no_hyphen_passthrough(service):
    assert service._tenant_key("plainid", "agent_availability") == "tenant:plainid:agent_availability"


# ---------------------------------------------------------------------------
# is_within_business_hours — 주간 영업
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_business_hours_within_range(service):
    service._redis.hget = AsyncMock(return_value="09:00-18:00")
    now = _kst(2026, 4, 20, 14, 0)  # Monday 14:00
    assert await service.is_within_business_hours("t1", now=now) is True


@pytest.mark.asyncio
async def test_business_hours_start_boundary_inclusive(service):
    service._redis.hget = AsyncMock(return_value="09:00-18:00")
    now = _kst(2026, 4, 20, 9, 0)
    assert await service.is_within_business_hours("t1", now=now) is True


@pytest.mark.asyncio
async def test_business_hours_end_boundary_inclusive(service):
    service._redis.hget = AsyncMock(return_value="09:00-18:00")
    now = _kst(2026, 4, 20, 18, 0)
    assert await service.is_within_business_hours("t1", now=now) is True


@pytest.mark.asyncio
async def test_business_hours_before_open(service):
    service._redis.hget = AsyncMock(return_value="09:00-18:00")
    now = _kst(2026, 4, 20, 8, 59)
    assert await service.is_within_business_hours("t1", now=now) is False


@pytest.mark.asyncio
async def test_business_hours_after_close(service):
    service._redis.hget = AsyncMock(return_value="09:00-18:00")
    now = _kst(2026, 4, 20, 18, 1)
    assert await service.is_within_business_hours("t1", now=now) is False


# ---------------------------------------------------------------------------
# is_within_business_hours — 휴무/빈 값/파싱 실패
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_business_hours_closed(service):
    service._redis.hget = AsyncMock(return_value="closed")
    now = _kst(2026, 4, 26, 14, 0)  # Sunday
    assert await service.is_within_business_hours("t1", now=now) is False


@pytest.mark.asyncio
async def test_business_hours_missing_key_returns_false(service):
    service._redis.hget = AsyncMock(return_value=None)
    now = _kst(2026, 4, 20, 14, 0)
    assert await service.is_within_business_hours("t1", now=now) is False


@pytest.mark.asyncio
async def test_business_hours_malformed_value(service):
    service._redis.hget = AsyncMock(return_value="not-a-time-range")
    now = _kst(2026, 4, 20, 14, 0)
    assert await service.is_within_business_hours("t1", now=now) is False


@pytest.mark.asyncio
async def test_business_hours_invalid_time_format(service):
    service._redis.hget = AsyncMock(return_value="9-18")
    now = _kst(2026, 4, 20, 14, 0)
    assert await service.is_within_business_hours("t1", now=now) is False


# ---------------------------------------------------------------------------
# is_within_business_hours — 오버나이트 (야간업종)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_business_hours_overnight_first_half(service):
    """22:00-02:00, 23:00 → True (시작일 쪽)."""
    service._redis.hget = AsyncMock(return_value="22:00-02:00")
    now = _kst(2026, 4, 20, 23, 0)
    assert await service.is_within_business_hours("t1", now=now) is True


@pytest.mark.asyncio
async def test_business_hours_overnight_second_half(service):
    """22:00-02:00, 01:00 → True (다음날 쪽).

    주의: weekday는 now 기준. 즉 01:00은 화요일이지만 화요일 hash 조회 결과도
    동일한 오버나이트 셋으로 설정돼있다고 가정하고 단순 시각 판정만 검증.
    """
    service._redis.hget = AsyncMock(return_value="22:00-02:00")
    now = _kst(2026, 4, 21, 1, 0)  # Tuesday 01:00
    assert await service.is_within_business_hours("t1", now=now) is True


@pytest.mark.asyncio
async def test_business_hours_overnight_closed_window(service):
    """22:00-02:00, 10:00 → False (야간업종의 낮 휴무)."""
    service._redis.hget = AsyncMock(return_value="22:00-02:00")
    now = _kst(2026, 4, 20, 10, 0)
    assert await service.is_within_business_hours("t1", now=now) is False


# ---------------------------------------------------------------------------
# is_within_business_hours — Redis 장애
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_business_hours_redis_exception_returns_false(service):
    service._redis.hget = AsyncMock(side_effect=ConnectionError("redis down"))
    now = _kst(2026, 4, 20, 14, 0)
    assert await service.is_within_business_hours("t1", now=now) is False


# ---------------------------------------------------------------------------
# is_within_business_hours — 요일 매핑
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_business_hours_queries_correct_weekday_field(service):
    """hget이 (key, weekday) 순서로 호출되는지 검증."""
    service._redis.hget = AsyncMock(return_value="09:00-18:00")
    now = _kst(2026, 4, 26, 12, 0)  # Sunday
    await service.is_within_business_hours("abc-def", now=now)
    call_args = service._redis.hget.call_args
    assert call_args.args[0] == "tenant:abcdef:business_hours"
    assert call_args.args[1] == "sun"


# ---------------------------------------------------------------------------
# get_available_agent_count
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_count_returns_int(service):
    service._redis.hget = AsyncMock(return_value="3")
    assert await service.get_available_agent_count("t1") == 3


@pytest.mark.asyncio
async def test_agent_count_zero(service):
    service._redis.hget = AsyncMock(return_value="0")
    assert await service.get_available_agent_count("t1") == 0


@pytest.mark.asyncio
async def test_agent_count_missing_key(service):
    service._redis.hget = AsyncMock(return_value=None)
    assert await service.get_available_agent_count("t1") == 0


@pytest.mark.asyncio
async def test_agent_count_non_numeric(service):
    service._redis.hget = AsyncMock(return_value="abc")
    assert await service.get_available_agent_count("t1") == 0


@pytest.mark.asyncio
async def test_agent_count_redis_exception(service):
    service._redis.hget = AsyncMock(side_effect=TimeoutError("redis slow"))
    assert await service.get_available_agent_count("t1") == 0
