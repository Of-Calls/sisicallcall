"""escalation_branch_node 유닛 테스트.

모듈 레벨 싱글톤 (_session_service, _sync_summary) 을 monkeypatch 로 교체.
검증:
 - offhours / callback / immediate 3분할 진입 조건
 - 각 분기의 response_text / response_path / is_timeout
 - immediate 경로에서만 Summary 동기 호출
 - Summary 타임아웃·예외 시에도 immediate 멘트 그대로 반환 (handoff 진행)
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from app.agents.conversational.nodes.escalation_branch_node import (
    escalation_branch_node as escalation_mod,
)


def _state() -> dict:
    return {
        "call_id": "call-esc-1",
        "tenant_id": "tenant-xyz",
    }


@pytest.fixture
def patched(monkeypatch):
    fake_session = AsyncMock()
    fake_summary = AsyncMock()
    monkeypatch.setattr(escalation_mod, "_session_service", fake_session)
    monkeypatch.setattr(escalation_mod, "_sync_summary", fake_summary)
    return fake_session, fake_summary


# ---------------------------------------------------------------------------
# offhours
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_offhours_returns_offhours_message(patched):
    fake_session, fake_summary = patched
    fake_session.is_within_business_hours = AsyncMock(return_value=False)
    fake_session.get_available_agent_count = AsyncMock()  # 호출 금지
    fake_summary.run = AsyncMock()  # 호출 금지

    result = await escalation_mod.escalation_branch_node(_state())

    assert result == {
        "response_text": escalation_mod.MSG_OFFHOURS,
        "response_path": "escalation",
        "is_timeout": False,
    }
    fake_session.get_available_agent_count.assert_not_called()
    fake_summary.run.assert_not_called()


# ---------------------------------------------------------------------------
# callback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_callback_when_no_available_agent(patched):
    fake_session, fake_summary = patched
    fake_session.is_within_business_hours = AsyncMock(return_value=True)
    fake_session.get_available_agent_count = AsyncMock(return_value=0)
    fake_summary.run = AsyncMock()

    result = await escalation_mod.escalation_branch_node(_state())

    assert result == {
        "response_text": escalation_mod.MSG_CALLBACK,
        "response_path": "escalation",
        "is_timeout": False,
    }
    fake_summary.run.assert_not_called()


@pytest.mark.asyncio
async def test_callback_when_negative_agent_count(patched):
    """방어적 케이스: 음수 값도 callback 으로 라우트."""
    fake_session, fake_summary = patched
    fake_session.is_within_business_hours = AsyncMock(return_value=True)
    fake_session.get_available_agent_count = AsyncMock(return_value=-1)
    fake_summary.run = AsyncMock()

    result = await escalation_mod.escalation_branch_node(_state())

    assert result["response_text"] == escalation_mod.MSG_CALLBACK
    fake_summary.run.assert_not_called()


# ---------------------------------------------------------------------------
# immediate (정상)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_immediate_triggers_sync_summary(patched):
    fake_session, fake_summary = patched
    fake_session.is_within_business_hours = AsyncMock(return_value=True)
    fake_session.get_available_agent_count = AsyncMock(return_value=3)
    fake_summary.run = AsyncMock(return_value={"summary_short": "요약 결과"})

    result = await escalation_mod.escalation_branch_node(_state())

    assert result == {
        "response_text": escalation_mod.MSG_IMMEDIATE,
        "response_path": "escalation",
        "is_timeout": False,
    }
    fake_summary.run.assert_awaited_once_with(call_id="call-esc-1", tenant_id="tenant-xyz")


# ---------------------------------------------------------------------------
# immediate + Summary 타임아웃 (handoff 계속)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_immediate_summary_timeout_still_returns_immediate(monkeypatch, patched):
    """Summary가 타임아웃되어도 immediate 멘트로 상담원 연결은 진행."""
    fake_session, fake_summary = patched
    fake_session.is_within_business_hours = AsyncMock(return_value=True)
    fake_session.get_available_agent_count = AsyncMock(return_value=1)

    async def slow_run(**_):
        await asyncio.sleep(1.0)

    fake_summary.run = slow_run
    # 타임아웃 상수를 0.01초로 낮춰 실제 wait_for 타임아웃 경로를 강제
    monkeypatch.setattr(escalation_mod, "SUMMARY_SYNC_TIMEOUT_SEC", 0.01)

    result = await escalation_mod.escalation_branch_node(_state())

    assert result["response_text"] == escalation_mod.MSG_IMMEDIATE
    assert result["response_path"] == "escalation"
    # 이 is_timeout은 '브랜치 전체' 레벨 타임아웃용이며,
    # Summary 타임아웃은 handoff_notes 비우기로 대응 — 브랜치 자체는 정상 반환.
    assert result["is_timeout"] is False


# ---------------------------------------------------------------------------
# immediate + Summary 예외 (handoff 계속)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_immediate_summary_exception_still_returns_immediate(patched):
    fake_session, fake_summary = patched
    fake_session.is_within_business_hours = AsyncMock(return_value=True)
    fake_session.get_available_agent_count = AsyncMock(return_value=2)
    fake_summary.run = AsyncMock(side_effect=RuntimeError("openai 500"))

    result = await escalation_mod.escalation_branch_node(_state())

    assert result["response_text"] == escalation_mod.MSG_IMMEDIATE
    assert result["response_path"] == "escalation"
    assert result["is_timeout"] is False
