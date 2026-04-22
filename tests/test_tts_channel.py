"""MockTTSOutputChannel 유닛 테스트 (RFC 001 v0.2 P1).

검증 목표:
 - lifecycle: open / push_stall / push_response / flush
 - 이중 stall 방지 (같은 턴 내 두 번째 push_stall 무시)
 - 서로 다른 call_id 간 격리
 - cancel 후 queue clear + stall_emitted 플래그는 유지
 - flush 후 같은 call_id 재 open 시 stall 재발동 허용
 - un-opened call_id 에 push 시 안전 처리 (경고 후 skip)
"""
import pytest

from app.services.tts.mock_channel import MockTTSOutputChannel


@pytest.fixture
def channel():
    return MockTTSOutputChannel()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_open_initializes_state(channel):
    await channel.open("call-1", "tenant-a")
    assert channel.is_open("call-1") is True
    assert channel.stall_emitted_for("call-1") is False
    assert channel.events_for("call-1") == []


@pytest.mark.asyncio
async def test_push_stall_then_response_order(channel):
    await channel.open("call-1", "tenant-a")
    await channel.push_stall("call-1", "잠시만요...", "general")
    await channel.push_response("call-1", "오전 9시부터 오후 6시입니다.", "faq")

    events = channel.events_for("call-1")
    assert len(events) == 2
    assert events[0]["type"] == "stall"
    assert events[0]["text"] == "잠시만요..."
    assert events[0]["metadata"]["audio_field"] == "general"
    assert events[1]["type"] == "response"
    assert events[1]["text"] == "오전 9시부터 오후 6시입니다."
    assert events[1]["metadata"]["response_path"] == "faq"


@pytest.mark.asyncio
async def test_flush_cleans_up_state(channel):
    await channel.open("call-1", "tenant-a")
    await channel.push_stall("call-1", "잠시만요...", "general")
    await channel.flush("call-1")

    assert channel.is_open("call-1") is False
    assert channel.stall_emitted_for("call-1") is False  # reset


# ---------------------------------------------------------------------------
# 이중 stall 방지 (RFC §6.8)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_double_stall_second_ignored(channel):
    """같은 턴 내 두 번째 push_stall 은 skip."""
    await channel.open("call-1", "tenant-a")
    await channel.push_stall("call-1", "first stall", "faq")
    await channel.push_stall("call-1", "second stall", "task")

    events = channel.events_for("call-1")
    assert len(events) == 1
    assert events[0]["text"] == "first stall"
    assert channel.stall_emitted_for("call-1") is True


@pytest.mark.asyncio
async def test_stall_after_flush_and_reopen_allowed(channel):
    """flush 후 같은 call_id 를 재 open 하면 stall 재방출 허용 (새 턴)."""
    await channel.open("call-1", "tenant-a")
    await channel.push_stall("call-1", "turn 1 stall", "general")
    await channel.flush("call-1")

    await channel.open("call-1", "tenant-a")
    await channel.push_stall("call-1", "turn 2 stall", "general")

    events = channel.events_for("call-1")
    # 이전 턴 emission 은 그대로 있고 (assert 편의), 새 턴 stall 이 append 됨
    assert events[-1]["text"] == "turn 2 stall"
    assert channel.stall_emitted_for("call-1") is True


# ---------------------------------------------------------------------------
# 서로 다른 call_id 격리
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stall_flag_isolated_between_calls(channel):
    await channel.open("call-1", "tenant-a")
    await channel.open("call-2", "tenant-a")

    await channel.push_stall("call-1", "stall for c1", "general")
    # call-2 는 아직 stall 안 함 — push 허용돼야
    await channel.push_stall("call-2", "stall for c2", "general")

    assert channel.stall_emitted_for("call-1") is True
    assert channel.stall_emitted_for("call-2") is True
    assert len(channel.events_for("call-1")) == 1
    assert len(channel.events_for("call-2")) == 1


@pytest.mark.asyncio
async def test_different_tenants_same_call_no_crosstalk(channel):
    """서로 다른 tenant 도 call_id 가 다르면 완전 독립."""
    await channel.open("call-a", "tenant-hospital")
    await channel.open("call-b", "tenant-restaurant")

    await channel.push_response("call-a", "병원 응답", "faq")
    await channel.push_response("call-b", "음식점 응답", "faq")

    assert channel.events_for("call-a")[0]["text"] == "병원 응답"
    assert channel.events_for("call-b")[0]["text"] == "음식점 응답"


# ---------------------------------------------------------------------------
# Cancel (barge-in)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_clears_queue(channel):
    await channel.open("call-1", "tenant-a")
    await channel.push_stall("call-1", "stall", "general")
    await channel.push_response("call-1", "response", "faq")
    assert len(channel.events_for("call-1")) == 2

    await channel.cancel("call-1")

    assert channel.events_for("call-1") == []
    assert channel.is_cancelled("call-1") is True


@pytest.mark.asyncio
async def test_cancel_preserves_stall_emitted_flag(channel):
    """RFC §6.7: cancel 은 stall_emitted 리셋하지 않음."""
    await channel.open("call-1", "tenant-a")
    await channel.push_stall("call-1", "stall", "general")
    await channel.cancel("call-1")

    assert channel.stall_emitted_for("call-1") is True  # 유지
    # 그러므로 같은 턴 내에서 다시 push_stall 해도 skip
    await channel.push_stall("call-1", "retry stall", "general")
    assert channel.events_for("call-1") == []  # 여전히 비어있음 (cancel 로 clear 됐고 재 stall 은 skip)


# ---------------------------------------------------------------------------
# Unopened call 방어
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_push_stall_on_unopened_call_is_noop(channel):
    await channel.push_stall("unknown-call", "stall", "general")
    assert channel.events_for("unknown-call") == []
    assert channel.is_open("unknown-call") is False


@pytest.mark.asyncio
async def test_push_response_on_unopened_call_is_noop(channel):
    await channel.push_response("unknown-call", "response", "faq")
    assert channel.events_for("unknown-call") == []


@pytest.mark.asyncio
async def test_cancel_on_unopened_call_is_noop(channel):
    await channel.cancel("unknown-call")
    assert channel.is_cancelled("unknown-call") is False


# ---------------------------------------------------------------------------
# 전체 시나리오 (integration-like)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_turn_scenario(channel):
    """실제 턴 흐름: 빠른 응답 (stall 안 나감)."""
    await channel.open("call-fast", "tenant-a")
    await channel.push_response("call-fast", "즉답", "cache")
    await channel.flush("call-fast")

    # push_response 만 했으니 stall_emitted=False 인 채로 flush 됨
    # events_for 는 flush 후에도 남아있음 (검증용)


@pytest.mark.asyncio
async def test_full_turn_with_stall(channel):
    """실제 턴 흐름: LLM 느려서 stall 먼저 나간 후 최종 응답."""
    await channel.open("call-slow", "tenant-a")
    await channel.push_stall("call-slow", "잠시만 기다려주세요", "faq")
    # ... 시뮬레이션: LLM 응답 오면
    await channel.push_response("call-slow", "오전 9시부터 영업합니다.", "faq")
    await channel.flush("call-slow")

    events = channel.events_for("call-slow")
    assert len(events) == 2
    assert events[0]["type"] == "stall"
    assert events[1]["type"] == "response"
