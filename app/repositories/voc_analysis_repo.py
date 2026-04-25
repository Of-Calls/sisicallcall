from __future__ import annotations
from app.utils.logger import get_logger

logger = get_logger(__name__)

_store: dict[str, dict] = {}


class VOCAnalysisRepository:
    async def save_voc_analysis(self, call_id: str, voc: dict) -> None:
        _store[call_id] = voc
        logger.debug("voc_analysis saved call_id=%s", call_id)

    async def get_voc_analysis(self, call_id: str) -> dict | None:
        return _store.get(call_id)
