from __future__ import annotations

import copy
from datetime import datetime, timezone

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── In-memory store ───────────────────────────────────────────────────────────

_voc_store: dict[str, dict] = {}   # call_id → voc record


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reset() -> None:
    """테스트 격리용."""
    _voc_store.clear()


# ── Module-level functions (KDT-77 interface) ─────────────────────────────────

async def save_voc_analysis(
    call_id: str,
    tenant_id: str,
    voc_analysis: dict,
) -> None:
    now = _now()
    existing = _voc_store.get(call_id, {})
    _voc_store[call_id] = {
        "call_id": call_id,
        "tenant_id": tenant_id,
        "voc_analysis": copy.deepcopy(voc_analysis),
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }
    logger.debug("voc_analysis saved call_id=%s", call_id)


async def get_voc_by_call_id(call_id: str) -> dict | None:
    record = _voc_store.get(call_id)
    return copy.deepcopy(record) if record is not None else None


# ── Backward-compatible class interface (used by save_result_node) ────────────

class VOCAnalysisRepository:
    async def save_voc_analysis(self, call_id: str, voc: dict) -> None:
        await save_voc_analysis(call_id=call_id, tenant_id="", voc_analysis=voc)

    async def get_voc_analysis(self, call_id: str) -> dict | None:
        record = await get_voc_by_call_id(call_id)
        return copy.deepcopy(record["voc_analysis"]) if record else None
