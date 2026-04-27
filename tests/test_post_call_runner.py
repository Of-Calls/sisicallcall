"""
KDT-79 prep: run_post_call_agent_safely 테스트.

runner.py 의 safe wrapper 가 예외를 외부로 전파하지 않으면서
PostCallAgent 의 결과를 올바르게 래핑하는지 검증한다.
"""
from __future__ import annotations

import pytest

from app.agents.post_call.runner import run_post_call_agent_safely


# ── 1. call_ended trigger → ok=True ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_runner_call_ended_ok():
    result = await run_post_call_agent_safely(
        call_id="runner-test-001",
        trigger="call_ended",
        tenant_id="test-tenant",
    )

    assert result["ok"] is True
    assert result["error"] is None
    assert isinstance(result["result"], dict)


# ── 2. escalation_immediate trigger → ok=True ────────────────────────────────

@pytest.mark.asyncio
async def test_runner_escalation_ok():
    result = await run_post_call_agent_safely(
        call_id="runner-test-002",
        trigger="escalation_immediate",
        tenant_id="test-tenant",
    )

    assert result["ok"] is True
    assert result["error"] is None
    r = result["result"]
    assert r["trigger"] == "escalation_immediate"


# ── 3. manual trigger → ok=True ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_runner_manual_ok():
    result = await run_post_call_agent_safely(
        call_id="runner-test-003",
        trigger="manual",
        tenant_id="default",
    )

    assert result["ok"] is True
    assert result["result"]["call_id"] == "runner-test-003"


# ── 4. 잘못된 trigger → ok=False, 예외 전파 없음 ─────────────────────────────

@pytest.mark.asyncio
async def test_runner_invalid_trigger_no_exception():
    result = await run_post_call_agent_safely(
        call_id="runner-test-004",
        trigger="completely_invalid_trigger_xyz",
        tenant_id="default",
    )

    # 예외가 밖으로 나오지 않아야 한다
    assert result["ok"] is False
    assert result["result"] is None
    assert result["error"] is not None
    assert isinstance(result["error"], str)


# ── 5. PostCallAgent.run 내부에서 예외 발생 → ok=False, 전파 없음 ────────────

@pytest.mark.asyncio
async def test_runner_internal_exception_no_propagation(monkeypatch):
    from app.agents.post_call import agent as agent_mod

    async def _boom(*args, **kwargs):
        raise RuntimeError("deliberate test explosion")

    monkeypatch.setattr(agent_mod.PostCallAgent, "run", _boom)

    result = await run_post_call_agent_safely(
        call_id="runner-test-005",
        trigger="manual",
    )

    assert result["ok"] is False
    assert result["result"] is None
    assert "deliberate test explosion" in result["error"]


# ── 6. 결과 dict 에 핵심 필드 포함 ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_runner_result_has_required_keys():
    result = await run_post_call_agent_safely(
        call_id="runner-test-006",
        trigger="manual",
        tenant_id="tenant-x",
    )

    assert result["ok"] is True
    r = result["result"]
    for key in ("call_id", "trigger", "partial_success", "errors"):
        assert key in r, f"result 에 {key!r} 가 없음"


# ── 7. 기본 파라미터 동작 ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_runner_default_params():
    """trigger, tenant_id 생략 시 기본값으로 실행되어야 한다."""
    result = await run_post_call_agent_safely(call_id="runner-test-007")

    assert result["ok"] is True
    r = result["result"]
    assert r["trigger"] == "call_ended"
