from __future__ import annotations

from app.services.mcp.base import BaseMCPService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GmailMCPService(BaseMCPService):
    service_name: str = "gmail"

    async def send_email(self, to: str, subject: str, body: str) -> bool:
        if self._use_real_mode():
            # TODO(M2): 실제 Gmail MCP SDK 호출 구현
            raise NotImplementedError("Gmail MCP 실제 연동 미구현 — MCP_GMAIL_REAL 해제 후 사용")
        logger.info("GmailMCPService(mock) send_email to=%s subject=%s", to, subject)
        return True
