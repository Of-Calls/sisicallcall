"""Tests for scripts/run_post_call_batch_from_db.py"""
from __future__ import annotations

import sys
import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── build_target_calls_sql ────────────────────────────────────────────────────

class TestBuildTargetCallsSql:
    def test_with_tenant_id_includes_filter(self):
        from scripts.run_post_call_batch_from_db import build_target_calls_sql

        sql, params = build_target_calls_sql(
            tenant_id="ba2bf499-6fcc-4340-b3dd-9341f8bcc915",
            limit=5,
            offset=0,
            only_missing=False,
        )

        assert "c.tenant_id = $1::uuid" in sql
        assert "ba2bf499-6fcc-4340-b3dd-9341f8bcc915" in params
        assert 5 in params

    def test_without_tenant_id_no_tenant_filter(self):
        from scripts.run_post_call_batch_from_db import build_target_calls_sql

        sql, params = build_target_calls_sql(
            tenant_id=None,
            limit=10,
            offset=0,
            only_missing=False,
        )

        # c.tenant_id appears in SELECT; the WHERE filter must NOT appear
        assert "AND c.tenant_id" not in sql
        # params should only contain limit and offset
        assert params == [10, 0]

    def test_only_missing_adds_having_clause(self):
        from scripts.run_post_call_batch_from_db import build_target_calls_sql

        sql, params = build_target_calls_sql(
            tenant_id=None,
            limit=5,
            offset=0,
            only_missing=True,
        )

        assert "HAVING" in sql
        assert "MAX(cs.call_id::text) IS NULL" in sql
        assert "MAX(va.call_id::text) IS NULL" in sql

    def test_without_only_missing_no_having(self):
        from scripts.run_post_call_batch_from_db import build_target_calls_sql

        sql, params = build_target_calls_sql(
            tenant_id=None,
            limit=5,
            offset=0,
            only_missing=False,
        )

        assert "HAVING" not in sql

    def test_tenant_id_and_only_missing_combined(self):
        from scripts.run_post_call_batch_from_db import build_target_calls_sql

        sql, params = build_target_calls_sql(
            tenant_id="tenant-uuid",
            limit=3,
            offset=2,
            only_missing=True,
        )

        assert "c.tenant_id = $1::uuid" in sql
        assert "HAVING" in sql
        assert params[0] == "tenant-uuid"
        assert 3 in params
        assert 2 in params

    def test_limit_and_offset_are_parameterized(self):
        from scripts.run_post_call_batch_from_db import build_target_calls_sql

        sql, params = build_target_calls_sql(
            tenant_id=None,
            limit=7,
            offset=14,
            only_missing=False,
        )

        assert "LIMIT" in sql
        assert "OFFSET" in sql
        assert 7 in params
        assert 14 in params


# ── run_batch ─────────────────────────────────────────────────────────────────

class TestRunBatch:
    @pytest.mark.asyncio
    async def test_transcript_count_zero_is_skipped(self):
        from scripts.run_post_call_batch_from_db import run_batch
        import app.agents.post_call.completed_call_runner as runner_mod

        calls = [
            {"call_id": "call-001", "tenant_id": "tenant-a", "transcript_count": 0},
        ]

        with patch.object(runner_mod, "run_post_call_for_completed_call") as mock_runner:
            results = await run_batch(calls=calls, trigger="call_ended", dry_run=False)

        assert len(results) == 1
        assert results[0]["status"] == "skip"
        assert results[0]["skip_reason"] == "transcripts_missing"
        mock_runner.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_does_not_call_runner(self):
        from scripts.run_post_call_batch_from_db import run_batch
        import app.agents.post_call.completed_call_runner as runner_mod

        calls = [
            {"call_id": "call-001", "tenant_id": "tenant-a", "transcript_count": 10},
            {"call_id": "call-002", "tenant_id": "tenant-a", "transcript_count": 5},
        ]

        with patch.object(runner_mod, "run_post_call_for_completed_call") as mock_runner:
            results = await run_batch(calls=calls, trigger="call_ended", dry_run=True)

        mock_runner.assert_not_called()
        assert all(r["status"] == "dry_run" for r in results)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_single_call_failure_continues_batch(self):
        from scripts.run_post_call_batch_from_db import run_batch
        import app.agents.post_call.completed_call_runner as runner_mod

        calls = [
            {"call_id": "call-001", "tenant_id": "tenant-a", "transcript_count": 5},
            {"call_id": "call-002", "tenant_id": "tenant-a", "transcript_count": 3},
        ]

        async def _mock_runner(call_id, tenant_id, trigger):
            if call_id == "call-001":
                return {"ok": False, "result": None, "error": "call_context_not_found"}
            return _ok_outcome()

        with patch.object(runner_mod, "run_post_call_for_completed_call", side_effect=_mock_runner):
            results = await run_batch(calls=calls, trigger="call_ended", dry_run=False)

        assert len(results) == 2
        assert results[0]["status"] == "fail"
        assert results[0]["error"] == "call_context_not_found"
        assert results[1]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_exception_in_call_continues_batch(self):
        from scripts.run_post_call_batch_from_db import run_batch
        import app.agents.post_call.completed_call_runner as runner_mod

        calls = [
            {"call_id": "call-001", "tenant_id": "tenant-a", "transcript_count": 5},
            {"call_id": "call-002", "tenant_id": "tenant-a", "transcript_count": 3},
        ]

        async def _mock_runner(call_id, tenant_id, trigger):
            if call_id == "call-001":
                raise RuntimeError("unexpected internal error")
            return _ok_outcome()

        with patch.object(runner_mod, "run_post_call_for_completed_call", side_effect=_mock_runner):
            results = await run_batch(calls=calls, trigger="call_ended", dry_run=False)

        assert len(results) == 2
        assert results[0]["status"] == "fail"
        assert "unexpected internal error" in results[0]["error"]
        assert results[1]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_transcript_skip_and_ok_mixed(self):
        from scripts.run_post_call_batch_from_db import run_batch
        import app.agents.post_call.completed_call_runner as runner_mod

        calls = [
            {"call_id": "call-001", "tenant_id": "tenant-a", "transcript_count": 0},
            {"call_id": "call-002", "tenant_id": "tenant-a", "transcript_count": 5},
        ]

        async def _mock_runner(call_id, tenant_id, trigger):
            return _ok_outcome()

        with patch.object(runner_mod, "run_post_call_for_completed_call", side_effect=_mock_runner):
            results = await run_batch(calls=calls, trigger="call_ended", dry_run=False)

        assert results[0]["status"] == "skip"
        assert results[1]["status"] == "ok"


# ── extract_call_result ───────────────────────────────────────────────────────

class TestExtractCallResult:
    def test_failed_outcome(self):
        from scripts.run_post_call_batch_from_db import extract_call_result

        result = extract_call_result(
            call_id="call-001",
            tenant_id="tenant-a",
            transcript_count=5,
            outcome={"ok": False, "result": None, "error": "call_context_not_found"},
        )

        assert result["status"] == "fail"
        assert result["error"] == "call_context_not_found"
        assert result["call_id"] == "call-001"

    def test_ok_outcome_extracts_all_fields(self):
        from scripts.run_post_call_batch_from_db import extract_call_result

        result = extract_call_result("call-001", "tenant-a", 10, _ok_outcome())

        assert result["status"] == "ok"
        assert result["review_verdict"] == "pass"
        assert result["review_confidence"] == 0.92
        assert result["review_confidence_source"] == "llm"
        assert result["primary_category"] == "예약/일정"
        assert result["customer_emotion"] == "neutral"
        assert result["resolution_status"] == "resolved"
        assert result["priority"] == "medium"
        assert result["sentiment"] == "neutral"
        assert result["action_success"] == 1
        assert result["action_skipped"] == 1
        assert result["action_failed"] == 0
        assert result["action_plan_count"] == 1
        assert result["executed_count"] == 2

    def test_empty_result_uses_fallback_values(self):
        from scripts.run_post_call_batch_from_db import extract_call_result

        outcome = {"ok": True, "result": {}, "error": None}
        result = extract_call_result("call-001", "tenant-a", 3, outcome)

        assert result["status"] == "ok"
        assert result["review_verdict"] == "—"
        assert result["primary_category"] == "—"
        assert result["customer_emotion"] == "—"
        assert result["action_plan_count"] == 0
        assert result["action_success"] == 0


# ── fetch_tenant_report SQL ───────────────────────────────────────────────────

class TestFetchTenantReportSql:
    @pytest.mark.asyncio
    async def test_all_queries_use_tenant_id(self):
        from scripts.run_post_call_batch_from_db import fetch_tenant_report

        tenant_id = "ba2bf499-6fcc-4340-b3dd-9341f8bcc915"
        captured: list[tuple] = []

        async def mock_fetchrow(sql, *params):
            captured.append(("fetchrow", params))
            return (0,)

        async def mock_fetch(sql, *params):
            captured.append(("fetch", params))
            return []

        mock_conn = MagicMock()
        mock_conn.fetchrow = mock_fetchrow
        mock_conn.fetch = mock_fetch

        await fetch_tenant_report(mock_conn, tenant_id)

        assert len(captured) > 0
        for entry in captured:
            params = entry[1]
            assert tenant_id in params, f"tenant_id not in params for entry: {entry}"

    @pytest.mark.asyncio
    async def test_action_log_mismatch_uses_text_cast(self):
        from scripts.run_post_call_batch_from_db import fetch_tenant_report

        captured_sqls: list[str] = []

        async def mock_fetchrow(sql, *params):
            captured_sqls.append(sql)
            return (0,)

        async def mock_fetch(sql, *params):
            captured_sqls.append(sql)
            return []

        mock_conn = MagicMock()
        mock_conn.fetchrow = mock_fetchrow
        mock_conn.fetch = mock_fetch

        await fetch_tenant_report(mock_conn, "test-tenant-id")

        action_log_sqls = [s for s in captured_sqls if "mcp_action_logs" in s]
        assert len(action_log_sqls) == 1, "mcp_action_logs query should appear exactly once"
        assert "c.tenant_id::text" in action_log_sqls[0]
        assert "$1::text" in action_log_sqls[0]

    @pytest.mark.asyncio
    async def test_report_returns_expected_keys(self):
        from scripts.run_post_call_batch_from_db import fetch_tenant_report

        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(return_value=(0,))
        mock_conn.fetch = AsyncMock(return_value=[])

        report = await fetch_tenant_report(mock_conn, "tenant-a")

        expected_keys = {
            "call_type", "emotion", "priority", "resolution",
            "missing_primary_category",
            "tenant_mismatch_summary",
            "tenant_mismatch_voc",
            "tenant_mismatch_action_logs",
        }
        assert expected_keys == set(report.keys())


# ── CLI — missing required group ─────────────────────────────────────────────

class TestCli:
    def test_no_tenant_option_raises_system_exit(self, monkeypatch):
        """Neither --tenant-id nor --all-tenants should produce an error."""
        monkeypatch.setattr(sys, "argv", ["batch_runner", "--limit", "5", "--llm-mode", "mock"])

        from scripts.run_post_call_batch_from_db import main

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code != 0

    def test_all_tenants_flag_accepted(self, monkeypatch):
        """--all-tenants should be accepted by argparse without a parse error.

        _main is patched to a no-op async function so the test never touches
        the DB and produces no RuntimeWarning about unawaited coroutines.
        """
        monkeypatch.setattr(
            sys, "argv",
            ["batch_runner", "--all-tenants", "--limit", "1", "--llm-mode", "mock", "--dry-run"],
        )

        from scripts import run_post_call_batch_from_db as batch_runner

        called = {"value": False}

        async def fake_main(**_kwargs):
            called["value"] = True

        monkeypatch.setattr(batch_runner, "_main", fake_main)

        # Must not raise SystemExit (which argparse raises on parse errors).
        batch_runner.main()

        assert called["value"] is True


# ── LLM mode override ────────────────────────────────────────────────────────

class TestLlmModeOverride:
    def test_cli_llm_mode_real_overrides_mock_env(self, monkeypatch):
        import scripts.run_post_call_from_db as db_runner
        import app.agents.post_call.llm_caller as llm_mod

        monkeypatch.setenv("POST_CALL_LLM_MODE", "mock")
        result = db_runner._apply_llm_mode("real")
        assert result == "real"
        assert llm_mod.get_post_call_llm_mode() == "real"

    def test_cli_llm_mode_mock_overrides_real_env(self, monkeypatch):
        import scripts.run_post_call_from_db as db_runner
        import app.agents.post_call.llm_caller as llm_mod

        monkeypatch.setenv("POST_CALL_LLM_MODE", "real")
        result = db_runner._apply_llm_mode("mock")
        assert result == "mock"
        assert llm_mod.get_post_call_llm_mode() == "mock"


# ── _collect_tenant_ids ───────────────────────────────────────────────────────

class TestCollectTenantIds:
    def test_single_tenant_returns_that_tenant(self):
        from scripts.run_post_call_batch_from_db import _collect_tenant_ids

        results = [
            {"tenant_id": "tid-a", "status": "ok"},
            {"tenant_id": "tid-a", "status": "ok"},
        ]
        ids = _collect_tenant_ids(tenant_id="tid-a", all_tenants=False, results=results)
        assert ids == ["tid-a"]

    def test_all_tenants_collects_unique_ordered(self):
        from scripts.run_post_call_batch_from_db import _collect_tenant_ids

        results = [
            {"tenant_id": "tid-b", "status": "ok"},
            {"tenant_id": "tid-a", "status": "ok"},
            {"tenant_id": "tid-b", "status": "fail"},
        ]
        ids = _collect_tenant_ids(tenant_id=None, all_tenants=True, results=results)
        assert ids == ["tid-b", "tid-a"]

    def test_empty_results_returns_empty(self):
        from scripts.run_post_call_batch_from_db import _collect_tenant_ids

        ids = _collect_tenant_ids(tenant_id=None, all_tenants=True, results=[])
        assert ids == []


# ── helpers ───────────────────────────────────────────────────────────────────

def _ok_outcome() -> dict:
    return {
        "ok": True,
        "result": {
            "review_verdict": "pass",
            "review_result": {
                "confidence": 0.92,
                "confidence_source": "llm",
                "corrected_keys": [],
            },
            "human_review_required": False,
            "executed_actions": [
                {"status": "success"},
                {"status": "skipped"},
            ],
            "action_plan": {"actions": [{"action_type": "create_task"}]},
            "summary": {
                "summary_short": "예약 문의",
                "customer_emotion": "neutral",
                "resolution_status": "resolved",
            },
            "priority_result": {"priority": "medium"},
            "voc_analysis": {
                "intent_result": {"primary_category": "예약/일정"},
                "sentiment_result": {"sentiment": "neutral"},
            },
            "partial_success": False,
        },
        "error": None,
    }
