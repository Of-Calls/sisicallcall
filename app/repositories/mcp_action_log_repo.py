from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── In-memory store ───────────────────────────────────────────────────────────

_DEFAULT_STORE_PATH = Path(".local/mcp_action_logs.json")
_action_store: dict[str, list[dict]] = {}   # call_id → [log_entry, ...]

_VALID_STATUSES = frozenset({"success", "failed", "skipped", "pending"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_store_path() -> Path:
    return Path(os.getenv("MCP_ACTION_LOG_FILE", str(_DEFAULT_STORE_PATH)))


def _load_store_from_file() -> dict[str, list[dict]]:
    path = _get_store_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("mcp_action_logs file load failed path=%s err=%s", path, exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("mcp_action_logs file ignored path=%s reason=not_dict", path)
        return {}
    store: dict[str, list[dict]] = {}
    for call_id, entries in raw.items():
        if isinstance(call_id, str) and isinstance(entries, list):
            store[call_id] = [
                copy.deepcopy(entry)
                for entry in entries
                if isinstance(entry, dict)
            ]
    return store


def _save_store_to_file(store: dict[str, list[dict]]) -> None:
    path = _get_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # TODO: replace local file mode with a DB-backed action log before production.
    # Local demo mode intentionally avoids file locking.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _load_into_memory() -> None:
    for call_id, entries in _load_store_from_file().items():
        _action_store[call_id] = entries


def _reset(remove_file: bool = False) -> None:
    """테스트 격리용."""
    _action_store.clear()
    if remove_file:
        path = _get_store_path()
        if path.exists():
            path.unlink()


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
    _load_into_memory()
    now = _now()
    entries = [
        _to_log_entry(a, call_id=call_id, tenant_id=tenant_id, now=now)
        for a in executed_actions
    ]
    _action_store.setdefault(call_id, []).extend(entries)
    _save_store_to_file(_action_store)
    logger.debug("action_logs saved call_id=%s count=%d", call_id, len(entries))


async def find_successful_action(
    call_id: str,
    action_type: str,
    tool: str,
) -> dict | None:
    _load_into_memory()
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
    _load_into_memory()
    entries = _action_store.get(call_id, [])
    return copy.deepcopy(entries)


async def get_action_logs(
    tenant_id: str | None = None,
    started_from: str | None = None,
    started_to: str | None = None,
) -> list[dict]:
    _load_into_memory()
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
