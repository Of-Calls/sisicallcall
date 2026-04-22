"""cache_node 유닛 테스트.

모듈 레벨 싱글톤(_embedding_service, _cache_service)을 monkeypatch로 교체한다.
검증 목표:
 - hit / miss 리턴 포맷
 - embed 실패 격리 (cache 조회 시도 안 함)
 - cache 조회 실패 격리 (embedding은 유지)
 - downstream이 쓸 query_embedding이 반드시 dict에 실려 나감 (langgraph_spec §4.4)
"""
from unittest.mock import AsyncMock

import pytest

from app.agents.conversational.nodes.cache_node import cache_node as cache_node_mod


def _state() -> dict:
    """cache_node가 읽는 필드만 최소로 채운 CallState dict."""
    return {
        "call_id": "call-123",
        "tenant_id": "tenant-abc",
        "normalized_text": "영업 시간 알려주세요",
    }


@pytest.fixture
def patched(monkeypatch):
    """embedding / cache 서비스를 AsyncMock으로 교체 후 핸들 반환."""
    fake_embed = AsyncMock()
    fake_cache = AsyncMock()
    monkeypatch.setattr(cache_node_mod, "_embedding_service", fake_embed)
    monkeypatch.setattr(cache_node_mod, "_cache_service", fake_cache)
    return fake_embed, fake_cache


# ---------------------------------------------------------------------------
# Happy path: hit / miss
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_hit_returns_response_and_path(patched):
    fake_embed, fake_cache = patched
    fake_embed.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    fake_cache.lookup = AsyncMock(return_value={"response_text": "오전 9시부터 오후 6시입니다"})

    result = await cache_node_mod.cache_node(_state())

    assert result["cache_hit"] is True
    assert result["response_text"] == "오전 9시부터 오후 6시입니다"
    assert result["response_path"] == "cache"
    assert result["query_embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_cache_miss_returns_embedding_without_response(patched):
    fake_embed, fake_cache = patched
    fake_embed.embed = AsyncMock(return_value=[0.4, 0.5])
    fake_cache.lookup = AsyncMock(return_value=None)

    result = await cache_node_mod.cache_node(_state())

    assert result["cache_hit"] is False
    assert result["query_embedding"] == [0.4, 0.5]
    # miss 시에는 response_text/response_path를 설정하지 않아야 다운스트림이 담당
    assert "response_text" not in result
    assert "response_path" not in result


# ---------------------------------------------------------------------------
# 실패 격리
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_failure_returns_empty_embedding_and_miss(patched):
    fake_embed, fake_cache = patched
    fake_embed.embed = AsyncMock(side_effect=RuntimeError("embed down"))
    fake_cache.lookup = AsyncMock()  # 호출되면 안 됨

    result = await cache_node_mod.cache_node(_state())

    assert result == {"query_embedding": [], "cache_hit": False}
    fake_cache.lookup.assert_not_called()


@pytest.mark.asyncio
async def test_cache_lookup_failure_preserves_embedding(patched):
    """cache 조회가 터져도 embedding은 downstream에 살아서 전달되어야 한다."""
    fake_embed, fake_cache = patched
    fake_embed.embed = AsyncMock(return_value=[0.7, 0.8, 0.9])
    fake_cache.lookup = AsyncMock(side_effect=TimeoutError("chroma slow"))

    result = await cache_node_mod.cache_node(_state())

    assert result["cache_hit"] is False
    assert result["query_embedding"] == [0.7, 0.8, 0.9]
    assert "response_text" not in result


# ---------------------------------------------------------------------------
# Interaction: 올바른 인자로 호출되는가
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_called_with_normalized_text(patched):
    fake_embed, fake_cache = patched
    fake_embed.embed = AsyncMock(return_value=[0.0])
    fake_cache.lookup = AsyncMock(return_value=None)

    await cache_node_mod.cache_node(_state())

    fake_embed.embed.assert_awaited_once_with("영업 시간 알려주세요")


@pytest.mark.asyncio
async def test_cache_lookup_called_with_text_and_tenant(patched):
    fake_embed, fake_cache = patched
    fake_embed.embed = AsyncMock(return_value=[0.0])
    fake_cache.lookup = AsyncMock(return_value=None)

    await cache_node_mod.cache_node(_state())

    fake_cache.lookup.assert_awaited_once_with(
        text="영업 시간 알려주세요",
        tenant_id="tenant-abc",
    )
