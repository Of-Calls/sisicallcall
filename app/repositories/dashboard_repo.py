from __future__ import annotations

import copy
from datetime import datetime, timezone

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── In-memory store ───────────────────────────────────────────────────────────

_dashboard_store: dict[str, dict] = {}   # call_id → stored record

_PRIORITY_ORDER: dict[str, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reset() -> None:
    """테스트 격리용."""
    _dashboard_store.clear()


def _filter_records(
    records: list[dict],
    tenant_id: str | None,
    started_from: str | None,
    started_to: str | None,
) -> list[dict]:
    if tenant_id is not None:
        records = [r for r in records if r.get("tenant_id") == tenant_id]
    if started_from is not None:
        records = [r for r in records if r.get("created_at", "") >= started_from]
    if started_to is not None:
        records = [r for r in records if r.get("created_at", "") <= started_to]
    return records


# ── Module-level functions (KDT-77 interface) ─────────────────────────────────

async def upsert_dashboard_payload(
    call_id: str,
    tenant_id: str,
    payload: dict,
) -> None:
    now = _now()
    existing = _dashboard_store.get(call_id, {})
    _dashboard_store[call_id] = {
        **copy.deepcopy(payload),
        "call_id": call_id,
        "tenant_id": tenant_id,
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }
    logger.debug("dashboard upserted call_id=%s", call_id)


async def get_dashboard_payload(call_id: str) -> dict | None:
    record = _dashboard_store.get(call_id)
    return copy.deepcopy(record) if record is not None else None


async def get_post_call_detail(call_id: str) -> dict:
    record = _dashboard_store.get(call_id, {})
    return copy.deepcopy({
        "summary": record.get("summary"),
        "voc_analysis": record.get("voc_analysis"),
        "priority_result": record.get("priority_result"),
        "action_plan": record.get("action_plan"),
        "executed_actions": record.get("executed_actions", []),
        "errors": record.get("errors", []),
        "partial_success": record.get("partial_success", False),
    })


async def get_dashboard_overview(
    tenant_id: str | None = None,
    started_from: str | None = None,
    started_to: str | None = None,
) -> dict:
    records = list(_dashboard_store.values())
    records = _filter_records(records, tenant_id, started_from, started_to)

    mcp_success = 0
    mcp_failed = 0
    for r in records:
        for action in r.get("executed_actions", []):
            status = action.get("status", "")
            if status == "success":
                mcp_success += 1
            elif status == "failed":
                mcp_failed += 1

    return {
        "total_calls": len(records),
        "resolved_count": sum(
            1 for r in records
            if (r.get("summary") or {}).get("resolution_status") == "resolved"
        ),
        "escalated_count": sum(
            1 for r in records
            if (
                (r.get("summary") or {}).get("resolution_status") == "escalated"
                or r.get("trigger") == "escalation_immediate"
            )
        ),
        "action_required_count": sum(
            1 for r in records
            if (
                (r.get("priority_result") or {}).get("action_required")
                or (r.get("action_plan") or {}).get("action_required")
            )
        ),
        "mcp_success_count": mcp_success,
        "mcp_failed_count": mcp_failed,
        "partial_success_count": sum(1 for r in records if r.get("partial_success")),
    }


async def get_emotion_distribution(
    tenant_id: str | None = None,
    started_from: str | None = None,
    started_to: str | None = None,
) -> dict:
    records = list(_dashboard_store.values())
    records = _filter_records(records, tenant_id, started_from, started_to)

    dist: dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0, "angry": 0}
    for r in records:
        emotion = (r.get("summary") or {}).get("customer_emotion", "neutral")
        if emotion in dist:
            dist[emotion] += 1
        else:
            dist["neutral"] += 1
    return dist


async def get_priority_queue(tenant_id: str | None = None) -> list[dict]:
    records = list(_dashboard_store.values())
    if tenant_id is not None:
        records = [r for r in records if r.get("tenant_id") == tenant_id]

    queue: list[dict] = []
    for r in records:
        pr = r.get("priority_result") or {}
        ap = r.get("action_plan") or {}
        priority = pr.get("priority") or pr.get("tier") or "low"
        action_required = bool(pr.get("action_required") or ap.get("action_required"))

        if priority in ("high", "critical") or action_required:
            summary = r.get("summary") or {}
            voc = r.get("voc_analysis") or {}
            intent = voc.get("intent_result") or {}
            queue.append({
                "call_id": r.get("call_id", ""),
                "tenant_id": r.get("tenant_id", ""),
                "priority": priority,
                "summary_short": summary.get("summary_short", ""),
                "primary_category": intent.get("primary_category", ""),
                "reason": pr.get("reason", ""),
                "created_at": r.get("created_at", ""),
            })

    queue.sort(key=lambda x: _PRIORITY_ORDER.get(x["priority"], 99))
    return copy.deepcopy(queue)


# ── Backward-compatible class interface (used by save_result_node) ────────────

class DashboardRepository:
    async def upsert_dashboard(self, call_id: str, payload: dict) -> None:
        tenant_id = payload.get("tenant_id", "")
        await upsert_dashboard_payload(call_id=call_id, tenant_id=tenant_id, payload=payload)

    async def get_dashboard(self, call_id: str) -> dict | None:
        return await get_dashboard_payload(call_id)

    async def list_dashboards(self, tenant_id: str) -> list[dict]:
        return [
            copy.deepcopy(v)
            for v in _dashboard_store.values()
            if v.get("tenant_id") == tenant_id
        ]
