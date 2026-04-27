"""
KDT-79 prep: context_provider 테스트.

get_call_context_for_post_call / seed_test_context 의 동작을 검증한다.
  - 주입된 컨텍스트 반환
  - 미주입 call_id → None
  - deepcopy 보호 (반환값 변경이 저장소를 오염하지 않음)
  - 필수 필드 구조 유지
"""
from __future__ import annotations

import pytest

import app.repositories.call_summary_repo as summary_mod

from app.agents.post_call.context_provider import (
    get_call_context_for_post_call,
    seed_test_context,
)


# ── 저장소 초기화 픽스처 ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_summary_store():
    summary_mod._reset()
    yield
    summary_mod._reset()


# ── 샘플 데이터 ──────────────────────────────────────────────────────────────

_SAMPLE_TRANSCRIPTS = [
    {"role": "customer", "text": "요금이 너무 비싸요"},
    {"role": "agent",    "text": "확인해 드리겠습니다"},
]

_SAMPLE_METADATA = {
    "call_id":   "ctx-test-001",
    "tenant_id": "tenant-test",
    "duration":  120,
}

_SAMPLE_BRANCH_STATS = {"faq": 2, "task": 1, "escalation": 0}


# ── 1. 주입된 컨텍스트가 반환된다 ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_provider_returns_seeded_context():
    await seed_test_context(
        call_id="ctx-test-001",
        tenant_id="tenant-test",
        transcripts=_SAMPLE_TRANSCRIPTS,
        call_metadata=_SAMPLE_METADATA,
        branch_stats=_SAMPLE_BRANCH_STATS,
    )

    ctx = await get_call_context_for_post_call("ctx-test-001", tenant_id="tenant-test")

    assert ctx is not None
    assert "transcripts" in ctx or "metadata" in ctx


# ── 2. 트랜스크립트 내용이 일치한다 ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_provider_transcripts_content():
    await seed_test_context(
        call_id="ctx-test-002",
        tenant_id="tenant-test",
        transcripts=_SAMPLE_TRANSCRIPTS,
    )

    ctx = await get_call_context_for_post_call("ctx-test-002")

    assert ctx is not None
    transcripts = ctx.get("transcripts", [])
    assert len(transcripts) == 2
    assert transcripts[0]["role"] == "customer"
    assert transcripts[1]["role"] == "agent"


# ── 3. 미주입 call_id → None ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_provider_unknown_call_id_returns_none():
    ctx = await get_call_context_for_post_call("definitely-does-not-exist-xyz")
    assert ctx is None


# ── 4. deepcopy 보호 — 반환값 변경이 저장소를 오염하지 않는다 ─────────────────

@pytest.mark.asyncio
async def test_context_provider_deepcopy_protection():
    await seed_test_context(
        call_id="ctx-test-004",
        tenant_id="tenant-test",
        transcripts=_SAMPLE_TRANSCRIPTS,
        call_metadata=_SAMPLE_METADATA,
    )

    ctx1 = await get_call_context_for_post_call("ctx-test-004")
    assert ctx1 is not None

    # 반환된 dict 를 변형한다
    if "transcripts" in ctx1:
        ctx1["transcripts"].append({"role": "agent", "text": "INJECTED"})
    ctx1["__extra__"] = "should_not_persist"

    # 다시 조회하면 원본이 유지되어야 한다
    ctx2 = await get_call_context_for_post_call("ctx-test-004")
    assert ctx2 is not None
    assert "__extra__" not in ctx2
    if "transcripts" in ctx2:
        assert len(ctx2["transcripts"]) == len(_SAMPLE_TRANSCRIPTS)


# ── 5. branch_stats 없이 주입해도 None 이 아니다 ─────────────────────────────

@pytest.mark.asyncio
async def test_context_provider_without_branch_stats():
    await seed_test_context(
        call_id="ctx-test-005",
        tenant_id="tenant-test",
        transcripts=_SAMPLE_TRANSCRIPTS,
        branch_stats=None,
    )

    ctx = await get_call_context_for_post_call("ctx-test-005")
    assert ctx is not None


# ── 6. metadata 만 있어도 컨텍스트를 반환한다 ────────────────────────────────

@pytest.mark.asyncio
async def test_context_provider_metadata_only():
    await seed_test_context(
        call_id="ctx-test-006",
        tenant_id="tenant-test",
        transcripts=[],        # 빈 transcript
        call_metadata=_SAMPLE_METADATA,
    )

    ctx = await get_call_context_for_post_call("ctx-test-006")
    assert ctx is not None
    assert ctx.get("metadata") is not None


# ── 7. 두 개의 다른 call_id 가 독립적으로 관리된다 ───────────────────────────

@pytest.mark.asyncio
async def test_context_provider_multiple_calls_isolated():
    await seed_test_context(
        call_id="ctx-test-007a",
        tenant_id="t-a",
        transcripts=[{"role": "customer", "text": "call A"}],
    )
    await seed_test_context(
        call_id="ctx-test-007b",
        tenant_id="t-b",
        transcripts=[{"role": "customer", "text": "call B"}],
    )

    ctx_a = await get_call_context_for_post_call("ctx-test-007a")
    ctx_b = await get_call_context_for_post_call("ctx-test-007b")

    assert ctx_a is not None
    assert ctx_b is not None

    # 각각 다른 데이터를 포함해야 한다
    t_a = ctx_a.get("transcripts", [])
    t_b = ctx_b.get("transcripts", [])
    if t_a and t_b:
        assert t_a[0]["text"] != t_b[0]["text"]
