"""Scenario A — escalation_branch_node against real Redis.

목적: Redis 키 네이밍 규약, HSET 프로토콜, decode_responses 인코딩,
요일 필드명 매핑을 실제 Redis 경유로 end-to-end 검증한다.
Unit test 의 AsyncMock 으로는 잡히지 않는 레이어.

실행:
  venv/Scripts/python -m pytest tests/integration/test_escalation_redis.py -v

사전 조건:
  make up 으로 redis 컨테이너 running + healthy.
"""
import uuid

import pytest
import pytest_asyncio

from app.agents.conversational.nodes.escalation_branch_node import (
    escalation_branch_node as escalation_mod,
)
from app.utils.config import settings

pytestmark = pytest.mark.integration

_ALL_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _hash_key(tenant_id: str, suffix: str) -> str:
    return f"tenant:{tenant_id.replace('-', '')}:{suffix}"


@pytest_asyncio.fixture(autouse=True)
async def _reset_module_singleton(monkeypatch):
    """모듈 레벨 _session_service 의 Redis client는 import 시점 event loop 에 바인딩된다.
    pytest-asyncio strict 모드는 테스트마다 loop 를 새로 만들므로, 2번째 테스트부터
    'Event loop is closed' 에러가 난다. 테스트마다 새 인스턴스로 리셋하여 이 loop 와
    바인딩된 client 를 사용하도록 한다. (프로덕션은 loop 하나라 문제 없음.)"""
    from app.services.session.redis_session import RedisSessionService
    fresh = RedisSessionService()
    monkeypatch.setattr(escalation_mod, "_session_service", fresh)
    yield
    try:
        await fresh._redis.aclose()
    except Exception:
        pass


@pytest_asyncio.fixture
async def redis_client():
    import redis.asyncio as aioredis
    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def tenant_id(redis_client):
    """테스트마다 고유 tenant_id 발급 + 종료 시 Redis 키 cleanup."""
    tid = f"integtest-{uuid.uuid4().hex[:12]}"
    yield tid
    await redis_client.delete(_hash_key(tid, "business_hours"))
    await redis_client.delete(_hash_key(tid, "agent_availability"))


async def _seed_all_open(redis_client, tenant_id):
    mapping = {day: "00:00-23:59" for day in _ALL_DAYS}
    await redis_client.hset(_hash_key(tenant_id, "business_hours"), mapping=mapping)


async def _seed_all_closed(redis_client, tenant_id):
    mapping = {day: "closed" for day in _ALL_DAYS}
    await redis_client.hset(_hash_key(tenant_id, "business_hours"), mapping=mapping)


async def _seed_agent_count(redis_client, tenant_id, count: int):
    await redis_client.hset(_hash_key(tenant_id, "agent_availability"), "available", str(count))


# ---------------------------------------------------------------------------
# offhours
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_offhours_via_real_redis(redis_client, tenant_id):
    await _seed_all_closed(redis_client, tenant_id)

    result = await escalation_mod.escalation_branch_node({
        "call_id": f"integ-{uuid.uuid4().hex[:6]}",
        "tenant_id": tenant_id,
    })

    assert result["response_text"] == escalation_mod.MSG_OFFHOURS
    assert result["response_path"] == "escalation"
    assert result["is_timeout"] is False


# ---------------------------------------------------------------------------
# callback (영업 중, 상담원 0명)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_callback_via_real_redis(redis_client, tenant_id):
    await _seed_all_open(redis_client, tenant_id)
    await _seed_agent_count(redis_client, tenant_id, 0)

    result = await escalation_mod.escalation_branch_node({
        "call_id": f"integ-{uuid.uuid4().hex[:6]}",
        "tenant_id": tenant_id,
    })

    assert result["response_text"] == escalation_mod.MSG_CALLBACK
    assert result["response_path"] == "escalation"


# ---------------------------------------------------------------------------
# immediate (영업 중, 상담원 가용)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_immediate_via_real_redis(redis_client, tenant_id):
    await _seed_all_open(redis_client, tenant_id)
    await _seed_agent_count(redis_client, tenant_id, 3)

    result = await escalation_mod.escalation_branch_node({
        "call_id": f"integ-{uuid.uuid4().hex[:6]}",
        "tenant_id": tenant_id,
    })

    assert result["response_text"] == escalation_mod.MSG_IMMEDIATE
    assert result["response_path"] == "escalation"
    # SyncSummaryAgent 는 현재 stub ({"summary_short": ""}) — 정상 반환되므로 is_timeout=False
    assert result["is_timeout"] is False


# ---------------------------------------------------------------------------
# 키가 아예 없을 때 → offhours 로 안전 fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_tenant_config_falls_back_to_offhours(redis_client, tenant_id):
    """테넌트 데이터가 Redis에 seed 되지 않은 상태 — 키 부재 시 False 경로."""
    # seed 없이 바로 호출
    result = await escalation_mod.escalation_branch_node({
        "call_id": f"integ-{uuid.uuid4().hex[:6]}",
        "tenant_id": tenant_id,
    })

    assert result["response_text"] == escalation_mod.MSG_OFFHOURS
