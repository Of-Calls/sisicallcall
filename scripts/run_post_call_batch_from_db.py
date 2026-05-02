"""Batch Post-call runner — DB에서 completed calls를 조회해 일괄 실행.

사용 예:
    python scripts/run_post_call_batch_from_db.py --tenant-id <uuid> --limit 5 --llm-mode mock
    python scripts/run_post_call_batch_from_db.py --tenant-id <uuid> --limit 2 --llm-mode real
    python scripts/run_post_call_batch_from_db.py --all-tenants --limit 10 --llm-mode mock
    python scripts/run_post_call_batch_from_db.py --tenant-id <uuid> --limit 5 --llm-mode mock --dry-run
    python scripts/run_post_call_batch_from_db.py --all-tenants --limit 10 --llm-mode mock --only-missing-results --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=False)

import asyncpg  # noqa: E402

from app.agents.post_call import completed_call_runner as runner_mod  # noqa: E402
from app.agents.post_call.llm_caller import (  # noqa: E402
    describe_post_call_llm,
    get_post_call_llm_mode,
    post_call_openai_key_available,
)
from app.utils.config import settings  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402
from scripts.run_post_call_demo import (  # noqa: E402
    _BOLD,
    _CYAN,
    _GREEN,
    _RED,
    _RESET,
    _YELLOW,
    _apply_connector_modes,
    _c,
    _patch_llm_nodes,
)
from scripts.run_post_call_from_db import (  # noqa: E402
    _apply_llm_mode,
    _patch_runner_context_lookup,
    _reset_llm_nodes,
)

logger = get_logger(__name__)

_SEP = "─" * 56


# ── DB helpers ────────────────────────────────────────────────────────────────

def _database_url() -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


# ── Target call query ────────────────────────────────────────────────────────

def build_target_calls_sql(
    tenant_id: str | None,
    limit: int,
    offset: int,
    only_missing: bool,
) -> tuple[str, list]:
    """Build SQL + params for querying completed calls.

    Returns a (sql, params) tuple suitable for asyncpg.fetch(*params).
    """
    sql = """
        SELECT
          c.id::text                              AS call_id,
          c.tenant_id::text                       AS tenant_id,
          c.twilio_call_sid,
          c.status,
          c.started_at,
          c.ended_at,
          COUNT(t.id)::int                        AS transcript_count,
          (MAX(cs.call_id::text) IS NOT NULL)     AS has_summary,
          (MAX(va.call_id::text) IS NOT NULL)     AS has_voc
        FROM calls c
        LEFT JOIN transcripts     t  ON t.call_id  = c.id
        LEFT JOIN call_summaries  cs ON cs.call_id = c.id
        LEFT JOIN voc_analyses    va ON va.call_id = c.id
        WHERE c.status = 'completed'"""

    params: list = []

    if tenant_id is not None:
        params.append(tenant_id)
        sql += f"\n  AND c.tenant_id = ${len(params)}::uuid"

    sql += "\nGROUP BY c.id, c.tenant_id, c.twilio_call_sid, c.status, c.started_at, c.ended_at"

    if only_missing:
        sql += "\nHAVING MAX(cs.call_id::text) IS NULL OR MAX(va.call_id::text) IS NULL"

    sql += "\nORDER BY COUNT(t.id) DESC"

    params.append(limit)
    sql += f"\nLIMIT ${len(params)}"
    params.append(offset)
    sql += f"\nOFFSET ${len(params)}"

    return sql, params


async def fetch_target_calls(
    conn,
    tenant_id: str | None,
    limit: int,
    offset: int,
    only_missing: bool,
) -> list[dict]:
    sql, params = build_target_calls_sql(tenant_id, limit, offset, only_missing)
    rows = await conn.fetch(sql, *params)
    return [dict(r) for r in rows]


# ── Per-call result extraction ────────────────────────────────────────────────

def extract_call_result(
    call_id: str,
    tenant_id: str,
    transcript_count: int,
    outcome: dict,
) -> dict:
    """Convert a runner outcome dict into a flat result record."""
    base = {
        "call_id": call_id,
        "tenant_id": str(tenant_id),
        "transcript_count": transcript_count,
    }
    if not outcome.get("ok"):
        return {**base, "status": "fail", "error": outcome.get("error")}

    result = outcome.get("result") or {}
    summary = result.get("summary") or {}
    priority_result = result.get("priority_result") or {}
    voc_analysis = result.get("voc_analysis") or {}
    intent_result = voc_analysis.get("intent_result") or {}
    sentiment_result = voc_analysis.get("sentiment_result") or {}
    review_result = result.get("review_result") or {}
    executed = result.get("executed_actions") or []
    actions = (result.get("action_plan") or {}).get("actions") or []

    return {
        **base,
        "status": "ok",
        "partial_success": result.get("partial_success", False),
        "review_verdict": result.get("review_verdict") or "—",
        "review_confidence": review_result.get("confidence"),
        "review_confidence_source": review_result.get("confidence_source"),
        "review_corrected_keys": review_result.get("corrected_keys"),
        "human_review_required": result.get("human_review_required", False),
        "action_plan_count": len(actions),
        "executed_count": len(executed),
        "action_success": sum(1 for a in executed if a.get("status") == "success"),
        "action_skipped": sum(1 for a in executed if a.get("status") == "skipped"),
        "action_failed": sum(1 for a in executed if a.get("status") == "failed"),
        "summary_short": summary.get("summary_short") or "—",
        "customer_emotion": summary.get("customer_emotion") or "—",
        "resolution_status": summary.get("resolution_status") or "—",
        "primary_category": intent_result.get("primary_category") or "—",
        "priority": priority_result.get("priority") or "—",
        "sentiment": sentiment_result.get("sentiment") or "—",
        "error": None,
    }


# ── Batch execution ──────────────────────────────────────────────────────────

async def run_batch(
    calls: list[dict],
    trigger: str,
    dry_run: bool = False,
) -> list[dict]:
    """Run post-call processing for each call in *calls*.

    - transcript_count == 0: skip (reason=transcripts_missing)
    - dry_run == True: record status=dry_run without calling the runner
    - Any single call exception is caught and recorded; the batch continues.
    - KeyboardInterrupt is re-raised immediately.
    """
    results: list[dict] = []

    for row in calls:
        call_id = str(row["call_id"])
        tenant_id = str(row["tenant_id"])
        transcript_count = int(row.get("transcript_count") or 0)

        if transcript_count == 0:
            results.append({
                "call_id": call_id,
                "tenant_id": tenant_id,
                "transcript_count": 0,
                "status": "skip",
                "skip_reason": "transcripts_missing",
            })
            continue

        if dry_run:
            results.append({
                "call_id": call_id,
                "tenant_id": tenant_id,
                "transcript_count": transcript_count,
                "status": "dry_run",
            })
            continue

        try:
            outcome = await runner_mod.run_post_call_for_completed_call(
                call_id=call_id,
                tenant_id=tenant_id,
                trigger=trigger,
            )
            results.append(extract_call_result(call_id, tenant_id, transcript_count, outcome))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.error("batch call failed call_id=%s err=%s", call_id, exc)
            results.append({
                "call_id": call_id,
                "tenant_id": tenant_id,
                "transcript_count": transcript_count,
                "status": "fail",
                "error": str(exc),
            })

    return results


# ── Report SQL queries ────────────────────────────────────────────────────────

async def fetch_tenant_report(conn, tenant_id: str) -> dict:
    """Query distribution metrics for a single tenant from the DB."""

    async def _count(sql: str, *params) -> int:
        row = await conn.fetchrow(sql, *params)
        return int(row[0]) if row else 0

    async def _dist(sql: str, *params) -> dict[str, int]:
        rows = await conn.fetch(sql, *params)
        return {str(r[0] or "—"): int(r[1]) for r in rows}

    call_type = await _dist(
        """
        SELECT intent_result->>'primary_category' AS call_type, COUNT(*)
        FROM voc_analyses
        WHERE tenant_id = $1::uuid
        GROUP BY 1 ORDER BY 2 DESC
        """,
        tenant_id,
    )

    emotion = await _dist(
        """
        SELECT customer_emotion, COUNT(*)
        FROM call_summaries
        WHERE tenant_id = $1::uuid
        GROUP BY 1 ORDER BY 2 DESC
        """,
        tenant_id,
    )

    priority = await _dist(
        """
        SELECT priority_result->>'priority' AS priority, COUNT(*)
        FROM voc_analyses
        WHERE tenant_id = $1::uuid
        GROUP BY 1 ORDER BY 2 DESC
        """,
        tenant_id,
    )

    resolution = await _dist(
        """
        SELECT resolution_status, COUNT(*)
        FROM call_summaries
        WHERE tenant_id = $1::uuid
        GROUP BY 1 ORDER BY 2 DESC
        """,
        tenant_id,
    )

    missing_category = await _count(
        """
        SELECT COUNT(*) FROM voc_analyses
        WHERE tenant_id = $1::uuid
          AND NULLIF(intent_result->>'primary_category', '') IS NULL
        """,
        tenant_id,
    )

    mismatch_summary = await _count(
        """
        SELECT COUNT(*) FROM call_summaries cs
        JOIN calls c ON c.id = cs.call_id
        WHERE cs.tenant_id <> c.tenant_id
          AND cs.tenant_id = $1::uuid
        """,
        tenant_id,
    )

    mismatch_voc = await _count(
        """
        SELECT COUNT(*) FROM voc_analyses va
        JOIN calls c ON c.id = va.call_id
        WHERE va.tenant_id <> c.tenant_id
          AND va.tenant_id = $1::uuid
        """,
        tenant_id,
    )

    mismatch_action_logs = await _count(
        """
        SELECT COUNT(*) FROM mcp_action_logs ml
        JOIN calls c ON c.id::text = ml.call_id
        WHERE ml.tenant_id IS DISTINCT FROM c.tenant_id::text
          AND c.tenant_id::text = $1::text
        """,
        tenant_id,
    )

    return {
        "call_type": call_type,
        "emotion": emotion,
        "priority": priority,
        "resolution": resolution,
        "missing_primary_category": missing_category,
        "tenant_mismatch_summary": mismatch_summary,
        "tenant_mismatch_voc": mismatch_voc,
        "tenant_mismatch_action_logs": mismatch_action_logs,
    }


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_sep(title: str = "") -> None:
    print()
    print(_SEP)
    if title:
        print(title)
        print(_SEP)


def _started_str(started_at) -> str:
    if started_at is None:
        return "—"
    if hasattr(started_at, "isoformat"):
        return started_at.isoformat()[:19]
    return str(started_at)[:19]


def _print_result_row(r: dict) -> None:
    call_id = r["call_id"]
    status = r["status"]

    if status == "skip":
        print(f"[SKIP] call_id={call_id}  reason={r.get('skip_reason', '—')}")
        return

    if status in ("fail", "dry_run"):
        tag = _c(_YELLOW, "DRY") if status == "dry_run" else _c(_RED, "FAIL")
        detail = "" if status == "dry_run" else f"  error={r.get('error', '—')}"
        print(f"[{tag}] call_id={call_id}{detail}")
        return

    conf = r.get("review_confidence")
    conf_str = f"{conf:.2f}" if isinstance(conf, float) else "—"
    verdict = r.get("review_verdict", "—")
    v_color = _GREEN if verdict == "pass" else (_YELLOW if verdict == "correctable" else _RED)

    print(
        f"[{_c(_GREEN, 'OK')}] "
        f"call_id={call_id}  "
        f"tenant={str(r['tenant_id'])[:8]}...  "
        f"category={r.get('primary_category', '—')}  "
        f"emotion={r.get('customer_emotion', '—')}  "
        f"priority={r.get('priority', '—')}  "
        f"review={_c(v_color, verdict)}  "
        f"confidence={conf_str}  "
        f"actions={r.get('action_plan_count', 0)}  "
        f"success={r.get('action_success', 0)}  "
        f"skipped={r.get('action_skipped', 0)}  "
        f"failed={r.get('action_failed', 0)}"
    )


def _print_dist(label: str, data: dict[str, int]) -> None:
    print(f"  {label}:")
    if data:
        for k, v in data.items():
            print(f"    {k}  {v}")
    else:
        print("    (데이터 없음)")


def _print_tenant_report(
    tenant_id: str,
    report: dict,
    ok_results: list[dict],
) -> None:
    print(f"\ntenant_id={tenant_id}")

    _print_dist("call_type", report["call_type"])
    _print_dist("emotion", report["emotion"])
    _print_dist("priority", report["priority"])
    _print_dist("resolution", report["resolution"])

    verdict_counts = Counter(
        r.get("review_verdict", "—")
        for r in ok_results
    )
    print("  review (이번 batch 기준):")
    if verdict_counts:
        for verdict, cnt in verdict_counts.items():
            print(f"    {verdict}  {cnt}")
    else:
        print("    (실행 결과 없음)")

    print("  data quality:")
    print(f"    missing_primary_category    {report['missing_primary_category']}")
    print(f"    tenant_mismatch_summary     {report['tenant_mismatch_summary']}")
    print(f"    tenant_mismatch_voc         {report['tenant_mismatch_voc']}")
    print(f"    tenant_mismatch_action_logs {report['tenant_mismatch_action_logs']}")


# ── Main async flow ───────────────────────────────────────────────────────────

async def _main(
    *,
    tenant_id: str | None,
    all_tenants: bool,
    limit: int,
    offset: int,
    llm_mode: str | None,
    dry_run: bool,
    only_missing_results: bool,
    trigger: str,
) -> None:
    effective_llm = _apply_llm_mode(llm_mode)
    if effective_llm == "real":
        _reset_llm_nodes()
    else:
        _patch_llm_nodes()

    _apply_connector_modes(real_actions=False, only_tool=None)
    _patch_runner_context_lookup()

    # ── Header ───────────────────────────────────────────────────────────────
    print("\nPost-call Batch Runner")
    llm_label = (
        _c(_GREEN, describe_post_call_llm())
        if effective_llm == "real"
        else _c(_YELLOW, "Demo Mock LLM")
    )
    print(f"  mode      : {llm_label}")
    if all_tenants:
        print(f"  tenant_id : (all tenants — 개발/운영자 진단용)")
    else:
        print(f"  tenant_id : {tenant_id}")
    print(f"  limit     : {limit}  offset : {offset}")
    if only_missing_results:
        print(f"  filter    : only-missing-results")
    if dry_run:
        print(f"\n  {_c(_YELLOW + _BOLD, 'DRY RUN: no post-call execution will be performed')}")
    if effective_llm == "real" and not post_call_openai_key_available():
        print(
            f"  LLM warn  : {_c(_YELLOW, 'OPENAI_API_KEY is missing; real LLM will fall back to mock')}"
        )

    # ── Query target calls ───────────────────────────────────────────────────
    try:
        conn = await asyncpg.connect(_database_url())
    except Exception as exc:
        print(f"\n{_c(_RED, 'DB connection failed:')} {exc}")
        sys.exit(1)

    try:
        calls = await fetch_target_calls(
            conn=conn,
            tenant_id=None if all_tenants else tenant_id,
            limit=limit,
            offset=offset,
            only_missing=only_missing_results,
        )
    finally:
        await conn.close()

    print(f"  targets   : {len(calls)}")

    # ── Print target list ────────────────────────────────────────────────────
    _print_sep("대상 calls")
    if not calls:
        print("  (대상 call 없음)")
    else:
        for i, row in enumerate(calls, 1):
            line = (
                f"  {i:3d}. call_id={row['call_id']}  "
                f"tenant={str(row['tenant_id'])[:8]}...  "
                f"transcripts={row['transcript_count']}  "
                f"started={_started_str(row.get('started_at'))}"
            )
            print(line)
            if dry_run:
                print(
                    f"       has_summary={row.get('has_summary', False)}  "
                    f"has_voc={row.get('has_voc', False)}  "
                    f"twilio_sid={row.get('twilio_call_sid') or '—'}"
                )

    if not calls:
        return

    # ── Execute (or dry-run) ─────────────────────────────────────────────────
    results = await run_batch(calls=calls, trigger=trigger, dry_run=dry_run)

    if dry_run:
        return

    ok_count = sum(1 for r in results if r["status"] == "ok")
    skip_count = sum(1 for r in results if r["status"] == "skip")
    fail_count = sum(1 for r in results if r["status"] == "fail")

    _print_sep("실행 결과")
    for r in results:
        _print_result_row(r)
    print(f"\n  합계: {ok_count} ok  {skip_count} skip  {fail_count} fail")

    # ── Tenant report ────────────────────────────────────────────────────────
    tenant_ids: list[str] = _collect_tenant_ids(tenant_id, all_tenants, results)
    if not tenant_ids:
        return

    _print_sep("tenant별 dashboard 원천 데이터 요약")

    try:
        conn = await asyncpg.connect(_database_url())
    except Exception as exc:
        print(f"\n{_c(_RED, 'DB connection failed for report:')} {exc}")
        return

    try:
        for tid in tenant_ids:
            ok_for_tenant = [
                r for r in results
                if r.get("tenant_id") == tid and r["status"] == "ok"
            ]
            try:
                report = await fetch_tenant_report(conn, tid)
                _print_tenant_report(tid, report, ok_for_tenant)
            except Exception as exc:
                print(f"  tenant_id={tid}  report_error={exc}")
    finally:
        await conn.close()


def _collect_tenant_ids(
    tenant_id: str | None,
    all_tenants: bool,
    results: list[dict],
) -> list[str]:
    if not all_tenants and tenant_id:
        return [tenant_id]
    seen: set[str] = set()
    ordered: list[str] = []
    for r in results:
        tid = r.get("tenant_id") or ""
        if tid and tid not in seen:
            seen.add(tid)
            ordered.append(tid)
    return ordered


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DB completed calls를 batch로 실행하고 tenant별 리포트를 출력한다.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tenant-id", help="특정 tenant의 completed call만 실행")
    group.add_argument(
        "--all-tenants",
        action="store_true",
        help="모든 tenant 대상 (개발/운영자 진단용; 출력은 tenant별로 구분)",
    )
    parser.add_argument("--limit",  type=int, default=5, help="실행 call 수 제한 (기본값 5)")
    parser.add_argument("--offset", type=int, default=0, help="조회 offset (기본값 0)")
    parser.add_argument(
        "--llm-mode",
        choices=["mock", "real"],
        default=None,
        help="POST_CALL_LLM_MODE 오버라이드. 기본값: env 설정 또는 mock",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 실행 없이 대상 call 목록만 출력",
    )
    parser.add_argument(
        "--only-missing-results",
        action="store_true",
        help="call_summaries 또는 voc_analyses가 없는 call만 대상",
    )
    parser.add_argument(
        "--trigger",
        default="call_ended",
        choices=["call_ended", "escalation_immediate", "manual"],
    )
    args = parser.parse_args()

    asyncio.run(
        _main(
            tenant_id=args.tenant_id,
            all_tenants=args.all_tenants,
            limit=args.limit,
            offset=args.offset,
            llm_mode=args.llm_mode,
            dry_run=args.dry_run,
            only_missing_results=args.only_missing_results,
            trigger=args.trigger,
        )
    )


if __name__ == "__main__":
    main()
