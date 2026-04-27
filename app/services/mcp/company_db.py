from __future__ import annotations

from app.services.mcp.base import BaseMCPService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CompanyDBMCPService(BaseMCPService):
    service_name: str = "company_db"

    async def query(self, sql: str, params: dict) -> list[dict]:
        if self._use_real_mode():
            raise NotImplementedError("CompanyDB MCP 실제 연동 미구현")
        logger.info("CompanyDBMCPService(mock) query sql=%s", sql)
        return []
