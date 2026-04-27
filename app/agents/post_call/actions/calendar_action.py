from __future__ import annotations

import os

from app.utils.logger import get_logger

logger = get_logger(__name__)


class CalendarAction:
    async def execute(self, action: dict, *, call_id: str, tenant_id: str = "") -> dict:
        params = action.get("params", {})
        logger.info("CalendarAction call_id=%s action_type=%s", call_id, action.get("action_type"))

        if os.getenv("MCP_CALENDAR_REAL", "").lower() in ("1", "true"):
            return await self._execute_real(params, call_id=call_id)

        event_id = f"calendar-mock-{call_id}"
        return {
            "external_id": event_id,
            "status": "success",
            "result": {
                "scheduled": True,
                "event_id": event_id,
                "title": params.get("title", "콜백 예약"),
                "mock": True,
            },
        }

    async def _execute_real(self, params: dict, *, call_id: str) -> dict:
        raise NotImplementedError("Calendar 실제 연동은 MCP 구성 후 구현")
