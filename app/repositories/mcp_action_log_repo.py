from __future__ import annotations

import copy
from datetime import datetime, timezone

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── In-memory store ───────────────────────────────────────────────────────────

_action_store: dict[str, list[dict]] = {}   # call_id → [log_entry, ...]

_VALID_STATUSES = frozenset({"success", "failed", "skipped", "pending"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reset() -> None:
    """테스트 격리용."""
    _action_store.clear()


def _to_log_entry(action: dict, *, call_id: str, tenant_id: str, now: str) -> dict:
    status = action.get("status", "pending")
    if status not in _VALID_STATUSES:
        status = "pending"
    return {
        "call_id": call_id,
        "tenant_id": tenant_id,
        "action_type": action.get("action_type", ""),
        "tool_name": action.get("tool", ""),
        "request_payload": copy.deepcopy(action.get("params", {})),
        "response_payload": copy.deepcopy(action.get("result", {})),
        "status": status,
        "external_id": action.get("external_id"),
        "error_message": action.get("error"),
        "created_at": now,
        "updated_at": now,
    }


# ── Module-level functions (KDT-77 interface) ─────────────────────────────────

async def save_action_logs(
    call_id: str,
    tenant_id: str,
    executed_actions: list[dict],
) -> None:
    now = _now()
    entries = [
        _to_log_entry(a, call_id=call_id, tenant_id=tenant_id, now=now)
        for a in executed_actions
    ]
    _action_store.setdefault(call_id, []).extend(entries)
    logger.debug("action_logs saved call_id=%s count=%d", call_id, len(entries))


async def find_successful_action(
    call_id: str,
    action_type: str,
    tool: str,
) -> dict | None:
    entries = _action_store.get(call_id, [])
    for entry in reversed(entries):
        if (
            entry.get("action_type") == action_type
            and entry.get("tool_name") == tool
            and entry.get("status") == "success"
        ):
            return copy.deepcopy(entry)
    return None


async def get_action_logs_by_call_id(call_id: str) -> list[dict]:
    entries = _action_store.get(call_id, [])
    return copy.deepcopy(entries)


async def get_action_logs(
    tenant_id: str | None = None,
    started_from: str | None = None,
    started_to: str | None = None,
) -> list[dict]:
    all_logs: list[dict] = []
    for entries in _action_store.values():
        all_logs.extend(copy.deepcopy(entries))

    if tenant_id is not None:
        all_logs = [e for e in all_logs if e.get("tenant_id") == tenant_id]
    if started_from is not None:
        all_logs = [e for e in all_logs if e.get("created_at", "") >= started_from]
    if started_to is not None:
        all_logs = [e for e in all_logs if e.get("created_at", "") <= started_to]
    return all_logs


# ── Backward-compatible class interface (used by save_result_node) ────────────

class MCPActionLogRepository:
    async def save_action_log(self, call_id: str, actions: list[dict]) -> None:
        await save_action_logs(call_id=call_id, tenant_id="", executed_actions=actions)
        logger.debug("action_log saved call_id=%s actions=%d", call_id, len(actions))

    async def get_action_log(self, call_id: str) -> list[dict]:
        return await get_action_logs_by_call_id(call_id)

    async def find_successful_action(
        self,
        call_id: str,
        action_type: str,
        tool: str,
    ) -> dict | None:
        return await find_successful_action(
            call_id=call_id,
            action_type=action_type,
            tool=tool,
        )
