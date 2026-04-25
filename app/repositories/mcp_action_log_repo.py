from __future__ import annotations
from app.utils.logger import get_logger

logger = get_logger(__name__)

_store: dict[str, list[dict]] = {}


class MCPActionLogRepository:
    async def save_action_log(self, call_id: str, actions: list[dict]) -> None:
        _store[call_id] = actions
        logger.debug("action_log saved call_id=%s actions=%d", call_id, len(actions))

    async def get_action_log(self, call_id: str) -> list[dict]:
        return _store.get(call_id, [])
