"""_run_with_stall 헬퍼 유닛 테스트 (RFC 001 v0.2 P2).

검증 목표:
 - Phase 1 빠른 응답: stall 없이 (response, False) 리턴
 - Phase 2 stall 발동 후 정상 응답: stall push + (response, False)
 - Phase 3 hardcut: stall push + (rag_results[0] or fallback, True)
 - 예외 처리: coro 가 raise 시 (fallback, False), 단계에 따라 stall 포함 여부
 - strip 처리: 응답 앞뒤 공백 제거
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

import app.agents.conversational.utils.stall as stall_mod
from app.services.tts.mock_channel import MockTTSOutputChannel


@pytest.fixture
def mock_channel(monkeypatch):
    ch = MockTTSOutputChannel()
    monkeypatch.setattr(stall_mod, "tts_channel", ch)
    return ch


async def _slow_return(text: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return text


async def _raise_after(delay: float, exc: Exception):
    await asyncio.sleep(delay)
    raise exc


async def _open(channel: MockTTSOutputChannel, call_id: str = "c1"):
    await channel.open(call_id, "tenant-a")


# ---------------------------------------------------------------------------
# Phase 1: 빠른 응답 (stall 없음)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fast_response_no_stall(mock_channel):
    await _open(mock_channel)

    response, is_timeout = await stall_mod._run_with_stall(
        coro=_slow_return("오전 9시부터", delay=0.01),
        call_id="c1",
        stall_msg="잠시만요...",
        stall_audio_field="faq",
        delay=0.5,
        hardcut_sec=3.0,
    )

    assert response == "오전 9시부터"
    assert is_timeout is False
    # stall 미발동 확인
    assert mock_channel.stall_emitted_for("c1") is False
    assert mock_channel.events_for("c1") == []


@pytest.mark.asyncio
async def test_response_is_stripped(mock_channel):
    await _open(mock_channel)

    response, _ = await stall_mod._run_with_stall(
        coro=_slow_return("  답변 내용  \n", delay=0.01),
        call_id="c1",
        stall_msg="stall",
        stall_audio_field="general",
        delay=0.5,
        hardcut_sec=3.0,
    )

    assert response == "답변 내용"


# ---------------------------------------------------------------------------
# Phase 2: stall 발동 후 정상 응답
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stall_triggered_then_success(mock_channel):
    await _open(mock_channel)

    # delay(0.05s) 넘기지만 hardcut(1.0s) 내 완료 — 총 0.15s 소요
    response, is_timeout = await stall_mod._run_with_stall(
        coro=_slow_return("실제 응답", delay=0.15),
        call_id="c1",
        stall_msg="잠시만요, 확인 중입니다",
        stall_audio_field="faq",
        delay=0.05,
        hardcut_sec=1.0,
    )

    assert response == "실제 응답"
    assert is_timeout is False
    # stall 이 Channel 에 push 됐어야 함
    assert mock_channel.stall_emitted_for("c1") is True
    events = mock_channel.events_for("c1")
    assert len(events) == 1
    assert events[0]["type"] == "stall"
    assert events[0]["text"] == "잠시만요, 확인 중입니다"
    assert events[0]["metadata"]["audio_field"] == "faq"


# ---------------------------------------------------------------------------
# Phase 3: hardcut 도달
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hardcut_with_rag_fallback(mock_channel):
    await _open(mock_channel)

    response, is_timeout = await stall_mod._run_with_stall(
        coro=_slow_return("너무 느림", delay=2.0),
        call_id="c1",
        stall_msg="stall",
        stall_audio_field="faq",
        delay=0.05,
        hardcut_sec=0.2,
        rag_results=["RAG 청크 1", "RAG 청크 2"],
    )

    assert response == "RAG 청크 1"
    assert is_timeout is True
    # stall 발동 확인
    assert mock_channel.stall_emitted_for("c1") is True


@pytest.mark.asyncio
async def test_hardcut_without_rag_uses_fallback_text(mock_channel):
    await _open(mock_channel)

    response, is_timeout = await stall_mod._run_with_stall(
        coro=_slow_return("느림", delay=2.0),
        call_id="c1",
        stall_msg="stall",
        stall_audio_field="general",
        delay=0.05,
        hardcut_sec=0.2,
        rag_results=None,
    )

    assert response == stall_mod.FALLBACK_MESSAGE
    assert is_timeout is True


@pytest.mark.asyncio
async def test_hardcut_with_empty_rag_uses_fallback_text(mock_channel):
    await _open(mock_channel)

    response, is_timeout = await stall_mod._run_with_stall(
        coro=_slow_return("느림", delay=2.0),
        call_id="c1",
        stall_msg="stall",
        stall_audio_field="faq",
        delay=0.05,
        hardcut_sec=0.2,
        rag_results=[],
    )

    assert response == stall_mod.FALLBACK_MESSAGE
    assert is_timeout is True


@pytest.mark.asyncio
async def test_custom_fallback_text(mock_channel):
    await _open(mock_channel)

    response, is_timeout = await stall_mod._run_with_stall(
        coro=_slow_return("느림", delay=2.0),
        call_id="c1",
        stall_msg="stall",
        stall_audio_field="general",
        delay=0.05,
        hardcut_sec=0.2,
        fallback_text="커스텀 폴백",
    )

    assert response == "커스텀 폴백"
    assert is_timeout is True


# ---------------------------------------------------------------------------
# 예외 처리
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phase1_exception_returns_fallback_no_stall(mock_channel):
    """Phase 1 중에 coro 가 raise → stall 안 나감, is_timeout=False."""
    await _open(mock_channel)

    response, is_timeout = await stall_mod._run_with_stall(
        coro=_raise_after(0.01, RuntimeError("openai 500")),
        call_id="c1",
        stall_msg="stall",
        stall_audio_field="faq",
        delay=0.5,
        hardcut_sec=3.0,
    )

    assert response == stall_mod.FALLBACK_MESSAGE
    assert is_timeout is False
    assert mock_channel.stall_emitted_for("c1") is False  # stall 미발동


@pytest.mark.asyncio
async def test_phase2_exception_returns_fallback_with_stall(mock_channel):
    """Phase 2 진입 후 (stall 발동 후) coro 가 raise → fallback 반환, stall 은 이미 Channel 에 있음."""
    await _open(mock_channel)

    response, is_timeout = await stall_mod._run_with_stall(
        coro=_raise_after(0.15, RuntimeError("timeout then error")),
        call_id="c1",
        stall_msg="잠시만요",
        stall_audio_field="faq",
        delay=0.05,
        hardcut_sec=1.0,
    )

    assert response == stall_mod.FALLBACK_MESSAGE
    assert is_timeout is False
    # stall 은 이미 Channel 에 push 된 상태
    assert mock_channel.stall_emitted_for("c1") is True
    events = mock_channel.events_for("c1")
    assert len(events) == 1
    assert events[0]["type"] == "stall"


# ---------------------------------------------------------------------------
# 격리 (다른 call_id)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_different_call_ids_isolated(mock_channel):
    """두 call_id 에 대해 동시 호출 시 stall push 가 각자 call_id 에 독립적으로 반영."""
    await _open(mock_channel, "c-a")
    await _open(mock_channel, "c-b")

    # c-a 만 stall 발동될 만큼 느림, c-b 는 빠름
    res_a, res_b = await asyncio.gather(
        stall_mod._run_with_stall(
            coro=_slow_return("a-response", delay=0.15),
            call_id="c-a",
            stall_msg="stall-a",
            stall_audio_field="faq",
            delay=0.05,
            hardcut_sec=1.0,
        ),
        stall_mod._run_with_stall(
            coro=_slow_return("b-response", delay=0.01),
            call_id="c-b",
            stall_msg="stall-b",
            stall_audio_field="faq",
            delay=0.05,
            hardcut_sec=1.0,
        ),
    )

    assert res_a == ("a-response", False)
    assert res_b == ("b-response", False)
    assert mock_channel.stall_emitted_for("c-a") is True
    assert mock_channel.stall_emitted_for("c-b") is False
