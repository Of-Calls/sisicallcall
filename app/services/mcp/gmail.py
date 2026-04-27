from __future__ import annotations

from app.services.mcp.base import BaseMCPService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GmailMCPService(BaseMCPService):
    service_name: str = "gmail"

    async def send_email(self, to: str, subject: str, body: str) -> bool:
        if self._use_real_mode():
            raise NotImplementedError("Gmail MCP 실제 연동 미구현")
        logger.info("GmailMCPService(mock) send_email to=%s subject=%s", to, subject)
        return True
