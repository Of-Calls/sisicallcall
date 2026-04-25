from __future__ import annotations
from app.utils.logger import get_logger

logger = get_logger(__name__)

_store: dict[str, dict] = {}


class DashboardRepository:
    async def upsert_dashboard(self, call_id: str, payload: dict) -> None:
        _store[call_id] = payload
        logger.debug("dashboard upserted call_id=%s", call_id)

    async def get_dashboard(self, call_id: str) -> dict | None:
        return _store.get(call_id)

    async def list_dashboards(self, tenant_id: str) -> list[dict]:
        return [v for v in _store.values() if v.get("tenant_id") == tenant_id]
