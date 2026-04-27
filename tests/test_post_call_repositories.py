"""
KDT-77 Repository 계층 테스트.

각 테스트는 모듈 레벨 _reset() 함수로 store를 초기화해
다른 테스트와 격리한다.
"""
from __future__ import annotations

import copy

import pytest

import app.repositories.call_summary_repo as summary_mod
import app.repositories.voc_analysis_repo as voc_mod
import app.repositories.mcp_action_log_repo as action_mod
import app.repositories.dashboard_repo as dashboard_mod

from app.repositories import (
    save_summary, get_summary_by_call_id,
    seed_call_context, get_call_context,
    save_voc_analysis, get_voc_by_call_id,
    save_action_logs, get_action_logs_by_call_id, get_action_logs,
    upsert_dashboard_payload, get_dashboard_payload,
    get_post_call_detail, get_dashboard_overview,
    get_emotion_distribution, get_priority_queue,
)
from tests.fixtures.sample_transcripts import (
    PAYLOAD_RESOLVED_NEUTRAL,
    PAYLOAD_ANGRY_CRITICAL,
    PAYLOAD_NEGATIVE_REPEATED,
    ALL_SAMPLE_PAYLOADS,
    TRANSCRIPTS_RESOLVED_NEUTRAL,
)


@pytest.fixture(autouse=True)
def reset_stores():
    """모든 in-memory store를 테스트 전후로 초기화한다."""
    summary_mod._reset()
    voc_mod._reset()
    action_mod._reset()
    dashboard_mod._reset()
    yield
    summary_mod._reset()
    voc_mod._reset()
    action_mod._reset()
    dashboard_mod._reset()


# ── 1. save_summary / get_summary_by_call_id ──────────────────────────────────

@pytest.mark.asyncio
async def test_save_and_get_summary():
    summary = {"summary_short": "테스트 요약", "customer_emotion": "neutral"}
    await save_summary("call-001", "tenant-x", summary)

    record = await get_summary_by_call_id("call-001")
    assert record is not None
    assert record["call_id"] == "call-001"
    assert record["tenant_id"] == "tenant-x"
    assert record["summary"]["summary_short"] == "테스트 요약"
    assert "created_at" in record
    assert "updated_at" in record


@pytest.mark.asyncio
async def test_get_summary_not_found():
    result = await get_summary_by_call_id("nonexistent")
    assert result is None


# ── 2. save_voc_analysis / get_voc_by_call_id ────────────────────────────────

@pytest.mark.asyncio
async def test_save_and_get_voc_analysis():
    voc = {"sentiment_result": {"sentiment": "negative", "intensity": 0.7, "reason": "불만"}}
    await save_voc_analysis("call-002", "tenant-x", voc)

    record = await get_voc_by_call_id("call-002")
    assert record is not None
    assert record["call_id"] == "call-002"
    assert record["tenant_id"] == "tenant-x"
    assert record["voc_analysis"]["sentiment_result"]["sentiment"] == "negative"
    assert "created_at" in record
    assert "updated_at" in record


@pytest.mark.asyncio
async def test_get_voc_not_found():
    result = await get_voc_by_call_id("nonexistent")
    assert result is None


# ── 3. save_action_logs / get_action_logs_by_call_id ─────────────────────────

@pytest.mark.asyncio
async def test_save_and_get_action_logs():
    actions = [
        {
            "action_type": "create_voc_issue",
            "tool": "company_db",
            "status": "success",
            "external_id": "VOC-MOCK-call-003",
            "error": None,
            "result": {"created": True},
            "params": {"tier": "high"},
        },
        {
            "action_type": "send_manager_email",
            "tool": "gmail",
            "status": "failed",
            "external_id": None,
            "error": "smtp error",
            "result": {},
            "params": {"to": "mgr@example.com"},
        },
    ]
    await save_action_logs("call-003", "tenant-x", actions)

    logs = await get_action_logs_by_call_id("call-003")
    assert len(logs) == 2
    assert logs[0]["action_type"] == "create_voc_issue"
    assert logs[0]["tool_name"] == "company_db"
    assert logs[0]["status"] == "success"
    assert logs[0]["request_payload"] == {"tier": "high"}
    assert logs[0]["response_payload"] == {"created": True}
    assert logs[0]["call_id"] == "call-003"
    assert logs[1]["status"] == "failed"
    assert logs[1]["error_message"] == "smtp error"


# ── 4. save_action_logs 재저장 시 기존 logs 교체 ──────────────────────────────

@pytest.mark.asyncio
async def test_save_action_logs_upsert_replaces():
    first = [{"action_type": "first_action", "tool": "company_db", "status": "success",
               "external_id": None, "error": None, "result": {}, "params": {}}]
    await save_action_logs("call-upsert", "tenant-x", first)

    second = [
        {"action_type": "second_action_a", "tool": "gmail", "status": "success",
         "external_id": None, "error": None, "result": {}, "params": {}},
        {"action_type": "second_action_b", "tool": "calendar", "status": "pending",
         "external_id": None, "error": None, "result": {}, "params": {}},
    ]
    await save_action_logs("call-upsert", "tenant-x", second)

    logs = await get_action_logs_by_call_id("call-upsert")
    assert len(logs) == 2
    assert logs[0]["action_type"] == "second_action_a"
    assert logs[1]["action_type"] == "second_action_b"


# ── 5. upsert_dashboard_payload / get_dashboard_payload ──────────────────────

@pytest.mark.asyncio
async def test_upsert_and_get_dashboard_payload():
    payload = copy.deepcopy(PAYLOAD_RESOLVED_NEUTRAL)
    await upsert_dashboard_payload("call-resolved-001", "tenant-a", payload)

    record = await get_dashboard_payload("call-resolved-001")
    assert record is not None
    assert record["call_id"] == "call-resolved-001"
    assert record["tenant_id"] == "tenant-a"
    assert "created_at" in record
    assert "updated_at" in record


@pytest.mark.asyncio
async def test_upsert_preserves_created_at():
    payload = copy.deepcopy(PAYLOAD_RESOLVED_NEUTRAL)
    await upsert_dashboard_payload("call-resolved-001", "tenant-a", payload)
    first = await get_dashboard_payload("call-resolved-001")
    first_created_at = first["created_at"]

    # 다시 upsert
    await upsert_dashboard_payload("call-resolved-001", "tenant-a", payload)
    second = await get_dashboard_payload("call-resolved-001")
    assert second["created_at"] == first_created_at  # created_at 불변


# ── 6. get_post_call_detail ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_post_call_detail():
    payload = copy.deepcopy(PAYLOAD_ANGRY_CRITICAL)
    await upsert_dashboard_payload("call-angry-002", "tenant-a", payload)

    detail = await get_post_call_detail("call-angry-002")
    assert "summary" in detail
    assert "voc_analysis" in detail
    assert "priority_result" in detail
    assert "action_plan" in detail
    assert "executed_actions" in detail
    assert "errors" in detail
    assert "partial_success" in detail
    assert detail["summary"]["customer_emotion"] == "angry"
    assert detail["priority_result"]["priority"] == "critical"


@pytest.mark.asyncio
async def test_get_post_call_detail_not_found_returns_safe_defaults():
    detail = await get_post_call_detail("nonexistent-call")
    assert detail["summary"] is None
    assert detail["executed_actions"] == []
    assert detail["errors"] == []
    assert detail["partial_success"] is False


# ── 7. get_dashboard_overview 집계 ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_dashboard_overview():
    for p in ALL_SAMPLE_PAYLOADS:
        await upsert_dashboard_payload(p["call_id"], p["tenant_id"], copy.deepcopy(p))

    overview = await get_dashboard_overview()
    assert overview["total_calls"] == 3
    # resolved_count: resolved_neutral(resolved) + negative_repeated(resolved) = 2
    assert overview["resolved_count"] == 2
    # escalated_count: angry_critical(trigger=escalation_immediate or resolution=escalated) = 1
    assert overview["escalated_count"] >= 1
    # action_required_count: angry(True) + negative(True) = 2
    assert overview["action_required_count"] == 2
    # mcp_success_count: angry(2 success) + negative(1 success + 1 failed) = 3
    assert overview["mcp_success_count"] == 3
    # mcp_failed_count: negative(1 failed) = 1
    assert overview["mcp_failed_count"] == 1
    # partial_success_count: negative_repeated = 1
    assert overview["partial_success_count"] == 1


# ── 8. get_emotion_distribution 집계 ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_emotion_distribution():
    for p in ALL_SAMPLE_PAYLOADS:
        await upsert_dashboard_payload(p["call_id"], p["tenant_id"], copy.deepcopy(p))

    dist = await get_emotion_distribution()
    assert dist["neutral"] == 1    # resolved_neutral
    assert dist["angry"] == 1      # angry_critical
    assert dist["negative"] == 1   # negative_repeated
    assert dist["positive"] == 0


# ── 9. get_priority_queue 정렬 ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_priority_queue_sorted():
    for p in ALL_SAMPLE_PAYLOADS:
        await upsert_dashboard_payload(p["call_id"], p["tenant_id"], copy.deepcopy(p))

    queue = await get_priority_queue()
    # resolved_neutral(low, action_required=False)는 제외
    assert all(
        q["priority"] in ("high", "critical") or q.get("action_required")
        for q in queue
    )
    # critical이 high보다 먼저
    priorities = [q["priority"] for q in queue]
    assert priorities.index("critical") < priorities.index("high")

    # 필수 필드 확인
    for item in queue:
        for key in ("call_id", "tenant_id", "priority", "summary_short", "primary_category",
                    "reason", "created_at"):
            assert key in item, f"priority_queue 항목에 {key!r} 키가 없음"


# ── 10. deepcopy 보장 ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_deepcopy_summary_not_contaminated():
    summary = {"summary_short": "원본", "customer_emotion": "neutral"}
    await save_summary("call-dc-001", "t", summary)

    record = await get_summary_by_call_id("call-dc-001")
    record["summary"]["summary_short"] = "오염된 값"

    # 재조회 시 원본 데이터 보존되어야 함
    fresh = await get_summary_by_call_id("call-dc-001")
    assert fresh["summary"]["summary_short"] == "원본"


@pytest.mark.asyncio
async def test_deepcopy_voc_not_contaminated():
    voc = {"sentiment_result": {"sentiment": "neutral"}}
    await save_voc_analysis("call-dc-002", "t", voc)

    record = await get_voc_by_call_id("call-dc-002")
    record["voc_analysis"]["sentiment_result"]["sentiment"] = "angry"

    fresh = await get_voc_by_call_id("call-dc-002")
    assert fresh["voc_analysis"]["sentiment_result"]["sentiment"] == "neutral"


@pytest.mark.asyncio
async def test_deepcopy_dashboard_not_contaminated():
    payload = copy.deepcopy(PAYLOAD_RESOLVED_NEUTRAL)
    await upsert_dashboard_payload("call-dc-003", "tenant-a", payload)

    record = await get_dashboard_payload("call-dc-003")
    record["summary"]["summary_short"] = "오염"

    fresh = await get_dashboard_payload("call-dc-003")
    assert fresh["summary"]["summary_short"] == PAYLOAD_RESOLVED_NEUTRAL["summary"]["summary_short"]


@pytest.mark.asyncio
async def test_deepcopy_action_logs_not_contaminated():
    actions = [{"action_type": "create_voc_issue", "tool": "company_db",
                "status": "success", "external_id": None, "error": None,
                "result": {"note": "원본"}, "params": {}}]
    await save_action_logs("call-dc-004", "t", actions)

    logs = await get_action_logs_by_call_id("call-dc-004")
    logs[0]["response_payload"]["note"] = "오염"

    fresh = await get_action_logs_by_call_id("call-dc-004")
    assert fresh[0]["response_payload"]["note"] == "원본"


# ── 11. tenant_id 필터 동작 ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dashboard_overview_tenant_filter():
    for p in ALL_SAMPLE_PAYLOADS:
        await upsert_dashboard_payload(p["call_id"], p["tenant_id"], copy.deepcopy(p))

    # tenant-a: resolved_neutral + angry_critical = 2
    overview_a = await get_dashboard_overview(tenant_id="tenant-a")
    assert overview_a["total_calls"] == 2

    # tenant-b: negative_repeated = 1
    overview_b = await get_dashboard_overview(tenant_id="tenant-b")
    assert overview_b["total_calls"] == 1


@pytest.mark.asyncio
async def test_action_logs_tenant_filter():
    actions_a = [{"action_type": "create_voc_issue", "tool": "company_db",
                  "status": "success", "external_id": None, "error": None,
                  "result": {}, "params": {}}]
    actions_b = [{"action_type": "send_manager_email", "tool": "gmail",
                  "status": "success", "external_id": None, "error": None,
                  "result": {}, "params": {}}]

    await save_action_logs("call-ta-001", "tenant-alpha", actions_a)
    await save_action_logs("call-ta-002", "tenant-beta", actions_b)

    logs_alpha = await get_action_logs(tenant_id="tenant-alpha")
    assert len(logs_alpha) == 1
    assert logs_alpha[0]["tenant_id"] == "tenant-alpha"

    logs_beta = await get_action_logs(tenant_id="tenant-beta")
    assert len(logs_beta) == 1


@pytest.mark.asyncio
async def test_priority_queue_tenant_filter():
    for p in ALL_SAMPLE_PAYLOADS:
        await upsert_dashboard_payload(p["call_id"], p["tenant_id"], copy.deepcopy(p))

    # tenant-a high/critical items: angry_critical(critical) = 1
    queue_a = await get_priority_queue(tenant_id="tenant-a")
    assert all(q["tenant_id"] == "tenant-a" for q in queue_a)

    # tenant-b: negative_repeated(high) = 1
    queue_b = await get_priority_queue(tenant_id="tenant-b")
    assert all(q["tenant_id"] == "tenant-b" for q in queue_b)


# ── 12. 빈 store에서도 기본값 안전 반환 ──────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_store_overview_safe_defaults():
    overview = await get_dashboard_overview()
    assert overview == {
        "total_calls": 0,
        "resolved_count": 0,
        "escalated_count": 0,
        "action_required_count": 0,
        "mcp_success_count": 0,
        "mcp_failed_count": 0,
        "partial_success_count": 0,
    }


@pytest.mark.asyncio
async def test_empty_store_emotion_distribution_safe_defaults():
    dist = await get_emotion_distribution()
    assert dist == {"positive": 0, "neutral": 0, "negative": 0, "angry": 0}


@pytest.mark.asyncio
async def test_empty_store_priority_queue_safe_defaults():
    queue = await get_priority_queue()
    assert queue == []


@pytest.mark.asyncio
async def test_empty_store_action_logs_safe_defaults():
    logs = await get_action_logs_by_call_id("no-such-call")
    assert logs == []


# ── seed_call_context + get_call_context ─────────────────────────────────────

@pytest.mark.asyncio
async def test_seed_and_get_call_context():
    await seed_call_context(
        call_id="seed-001",
        tenant_id="t",
        transcripts=TRANSCRIPTS_RESOLVED_NEUTRAL,
        call_metadata={"start_time": "2026-04-27T10:00:00Z"},
        branch_stats={"faq": 2},
    )
    ctx = await get_call_context("seed-001")
    assert ctx["transcripts"] == TRANSCRIPTS_RESOLVED_NEUTRAL
    assert ctx["metadata"]["call_id"] == "seed-001"
    assert ctx["metadata"]["tenant_id"] == "t"
    assert ctx["branch_stats"] == {"faq": 2}


@pytest.mark.asyncio
async def test_get_call_context_fallback_sample():
    """context가 없으면 sample context를 call_id 패치하여 반환한다."""
    ctx = await get_call_context("unknown-call")
    assert ctx is not None
    assert ctx["metadata"]["call_id"] == "unknown-call"
    assert len(ctx["transcripts"]) > 0
