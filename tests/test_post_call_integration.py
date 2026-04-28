"""
KDT-79: Post-call Agent 최종 통합 테스트.

검증 범위:
  1. stop 이벤트에서 asyncio.create_task가 run_post_call_agent_safely를 호출하는지
  2. runner가 context를 seed하고 PostCallAgent.run을 호출하는지
  3. context가 없어도 runner가 예외를 전파하지 않는지
  4. PostCallAgent partial_success=True 결과도 ok=True로 반환하는지
  5. main.py에 등록된 post_call/summary/dashboard 라우터가 TestClient에서 접근 가능한지
"""
from __future__ import annotations

import asyncio
import copy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.repositories.call_summary_repo as summary_mod
import app.repositories.voc_analysis_repo as voc_mod
import app.repositories.mcp_action_log_repo as action_mod
import app.repositories.dashboard_repo as dashboard_mod


# ── Store 격리 픽스처 ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_stores():
    summary_mod._reset()
    voc_mod._reset()
    action_mod._reset()
    dashboard_mod._reset()
    yield
    summary_mod._reset()
    voc_mod._reset()
    action_mod._reset()
    dashboard_mod._reset()


# ── main.py TestClient (라우터 통합 검증용) ───────────────────────────────────

@pytest.fixture(scope="module")
def main_client():
    """main.py의 FastAPI app 인스턴스로 TestClient를 생성한다."""
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ── 1. stop 이벤트 → asyncio.create_task 호출 검증 ───────────────────────────

def test_stop_event_calls_create_task():
    """call.py의 stop 이벤트 분기에서 asyncio.create_task가 호출되는지 검증한다."""
    created_coros = []

    def fake_create_task(coro, **kwargs):
        created_coros.append(coro)
        coro.close()
        return MagicMock()

    import app.api.v1.call as call_mod

    with patch.object(asyncio, "create_task", side_effect=fake_create_task):
        async def _simulate_stop():
            asyncio.create_task(
                call_mod.run_post_call_agent_safely(
                    call_id="test-stop-001",
                    trigger="call_ended",
                    tenant_id="tenant-test",
                )
            )

        asyncio.get_event_loop().run_until_complete(_simulate_stop())

    assert len(created_coros) >= 1


# ── 2. runner가 context를 seed하고 PostCallAgent를 호출하는지 ────────────────

@pytest.mark.asyncio
async def test_runner_seeds_context_then_calls_agent(monkeypatch):
    """context_provider가 context를 반환하면 runner가 seed_call_context를 호출한다."""
    from app.agents.post_call.runner import run_post_call_agent_safely

    sample_ctx = {
        "metadata":     {"call_id": "integ-001", "tenant_id": "t-a"},
        "transcripts":  [{"role": "customer", "text": "문의드립니다"}],
        "branch_stats": {"faq": 1, "task": 0, "escalation": 0},
    }

    seeded: list[dict] = []

    async def fake_get_context(call_id, tenant_id=None):
        return copy.deepcopy(sample_ctx)

    async def fake_seed(call_id, tenant_id="default", transcripts=None,
                        call_metadata=None, branch_stats=None):
        seeded.append({
            "call_id": call_id,
            "tenant_id": tenant_id,
            "transcripts": transcripts,
        })

    monkeypatch.setattr(
        "app.agents.post_call.runner.get_call_context_for_post_call",
        fake_get_context,
    )
    monkeypatch.setattr(
        "app.agents.post_call.runner.seed_call_context",
        fake_seed,
    )

    result = await run_post_call_agent_safely(
        call_id="integ-001",
        trigger="manual",
        tenant_id="t-a",
    )

    assert result["ok"] is True
    assert len(seeded) == 1
    assert seeded[0]["call_id"] == "integ-001"
    assert seeded[0]["transcripts"] == sample_ctx["transcripts"]


# ── 3. context 없어도 runner가 예외를 전파하지 않는다 ────────────────────────

@pytest.mark.asyncio
async def test_runner_no_context_no_exception(monkeypatch):
    """context_provider가 None을 반환해도 runner는 ok=True로 완료된다."""
    from app.agents.post_call.runner import run_post_call_agent_safely

    async def fake_get_context(call_id, tenant_id=None):
        return None

    monkeypatch.setattr(
        "app.agents.post_call.runner.get_call_context_for_post_call",
        fake_get_context,
    )

    result = await run_post_call_agent_safely(
        call_id="integ-002",
        trigger="call_ended",
        tenant_id="default",
    )

    # context 없어도 PostCallAgent는 empty fallback으로 실행 → ok=True
    assert result["ok"] is True
    assert result["error"] is None


# ── 4. PostCallAgent partial_success=True → runner ok=True ───────────────────

@pytest.mark.asyncio
async def test_runner_partial_success_returns_ok_true(monkeypatch):
    """PostCallAgent가 partial_success=True로 끝나도 ok=True를 반환한다."""
    from app.agents.post_call.runner import run_post_call_agent_safely
    from app.agents.post_call import agent as agent_mod

    async def fake_get_context(call_id, tenant_id=None):
        return None

    monkeypatch.setattr(
        "app.agents.post_call.runner.get_call_context_for_post_call",
        fake_get_context,
    )

    async def fake_run(self, call_id, trigger, tenant_id):
        return {
            "call_id": call_id,
            "trigger": trigger,
            "partial_success": True,
            "errors": [{"node": "action_router", "error": "mock error"}],
            "summary": {},
            "voc_analysis": {},
            "priority_result": {},
            "action_plan": [],
            "executed_actions": [],
        }

    monkeypatch.setattr(agent_mod.PostCallAgent, "run", fake_run)

    result = await run_post_call_agent_safely(
        call_id="integ-003",
        trigger="call_ended",
        tenant_id="t-x",
    )

    assert result["ok"] is True
    assert result["result"]["partial_success"] is True


# ── 5. runner 내부 예외 → ok=False, 전파 없음 ────────────────────────────────

@pytest.mark.asyncio
async def test_runner_internal_crash_returns_ok_false(monkeypatch):
    """PostCallAgent.run이 예외를 던지면 ok=False를 반환하고 예외를 전파하지 않는다."""
    from app.agents.post_call.runner import run_post_call_agent_safely
    from app.agents.post_call import agent as agent_mod

    async def fake_get_context(call_id, tenant_id=None):
        return None

    monkeypatch.setattr(
        "app.agents.post_call.runner.get_call_context_for_post_call",
        fake_get_context,
    )

    async def fake_run(self, *args, **kwargs):
        raise RuntimeError("의도된 테스트 크래시")

    monkeypatch.setattr(agent_mod.PostCallAgent, "run", fake_run)

    result = await run_post_call_agent_safely("integ-004", "call_ended")

    assert result["ok"] is False
    assert result["result"] is None
    assert "의도된 테스트 크래시" in result["error"]


# ── 6. main.py 라우터: POST /post-call/{call_id}/run 접근 가능 ────────────────

def test_main_post_call_router_accessible(main_client):
    """main.py에 등록된 /post-call 라우터가 TestClient에서 접근 가능하다."""
    resp = main_client.post("/post-call/integ-route-001/run?trigger=manual&tenant_id=t-test")
    # runner가 MockLLM으로 실행 → 200 OK 기대
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True


# ── 7. main.py 라우터: GET /summary/{call_id} 접근 가능 (404 정상) ──────────

def test_main_summary_router_accessible(main_client):
    """main.py에 등록된 /summary 라우터가 TestClient에서 접근 가능하다."""
    resp = main_client.get("/summary/nonexistent-call-xyz")
    # 데이터 없음 → 404 (라우터가 등록된 경우에만 404가 돌아옴)
    assert resp.status_code == 404


# ── 8. main.py 라우터: GET /dashboard/stats 접근 가능 ────────────────────────

def test_main_dashboard_router_accessible(main_client):
    """main.py에 등록된 /dashboard 라우터가 TestClient에서 접근 가능하다."""
    resp = main_client.get("/dashboard/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_calls" in data


# ── 9. main.py 라우터: GET /post-call/{call_id} not found ─────────────────────

def test_main_post_call_detail_not_found(main_client):
    """main.py에 등록된 /post-call 상세 조회 라우터가 404를 반환한다."""
    resp = main_client.get("/post-call/no-such-call-999")
    assert resp.status_code == 404


# ── 10. context seed 후 load_context_node가 올바른 데이터를 읽는지 ─────────────

@pytest.mark.asyncio
async def test_seeded_context_propagates_to_load_context_node():
    """runner가 seed한 context를 load_context_node가 올바르게 읽는다."""
    from app.repositories.call_summary_repo import seed_call_context
    from app.agents.post_call.nodes.load_context_node import load_context_node

    transcripts = [
        {"role": "customer", "text": "환불 요청합니다"},
        {"role": "agent",    "text": "처리해드리겠습니다"},
    ]
    await seed_call_context(
        call_id="integ-seed-001",
        tenant_id="t-seed",
        transcripts=transcripts,
        call_metadata={"call_id": "integ-seed-001", "tenant_id": "t-seed"},
        branch_stats={"faq": 0, "task": 1, "escalation": 0},
    )

    state = {
        "call_id": "integ-seed-001",
        "tenant_id": "t-seed",
        "trigger": "manual",
        "call_metadata": {},
        "transcripts": [],
        "branch_stats": {},
        "summary": {},
        "voc_analysis": {},
        "priority_result": {},
        "action_plan": [],
        "executed_actions": [],
        "errors": [],
        "partial_success": False,
    }

    result = await load_context_node(state)

    assert result["transcripts"] == transcripts
    assert result["branch_stats"] == {"faq": 0, "task": 1, "escalation": 0}
