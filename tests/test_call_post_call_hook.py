from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_completed_post_call_background_calls_completed_runner(monkeypatch):
    from app.agents.post_call import trigger

    called: list[dict] = []

    async def fake_runner(
        call_id: str,
        tenant_id: str = "default",
        trigger: str = "call_ended",
    ):
        called.append({"call_id": call_id, "tenant_id": tenant_id, "trigger": trigger})
        return {"ok": True, "result": {}, "error": None}

    monkeypatch.setattr(trigger, "run_post_call_for_completed_call", fake_runner)

    await trigger.run_completed_post_call_background("CA123", "tenant-1")

    assert called == [
        {"call_id": "CA123", "tenant_id": "tenant-1", "trigger": "call_ended"}
    ]


@pytest.mark.asyncio
async def test_completed_post_call_background_swallows_exceptions(monkeypatch):
    from app.agents.post_call import trigger

    async def fake_runner(
        call_id: str,
        tenant_id: str = "default",
        trigger: str = "call_ended",
    ):
        raise RuntimeError("runner crash")

    monkeypatch.setattr(trigger, "run_post_call_for_completed_call", fake_runner)

    await trigger.run_completed_post_call_background("CA456", "tenant-2")
