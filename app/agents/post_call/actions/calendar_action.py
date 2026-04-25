from __future__ import annotations
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CalendarAction:
    async def execute(self, action: dict, *, call_id: str) -> dict:
        params = action.get("params", {})
        logger.info("CalendarAction(dummy) call_id=%s action_type=%s", call_id, action.get("action_type"))
        # TODO: 실제 Calendar MCP 연동으로 교체
        return {
            "scheduled": True,
            "event_id": f"CAL-MOCK-{call_id}",
            "title": params.get("title", "콜백 예약"),
            "mock": True,
        }
