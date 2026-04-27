from __future__ import annotations

from app.services.mcp.base import BaseMCPService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CalendarMCPService(BaseMCPService):
    service_name: str = "calendar"

    async def create_event(self, title: str, start: str, end: str) -> dict:
        if self._use_real_mode():
            raise NotImplementedError("Calendar MCP 실제 연동 미구현")
        logger.info("CalendarMCPService(mock) create_event title=%s", title)
        return {"event_id": "mock-event", "title": title, "mock": True}
