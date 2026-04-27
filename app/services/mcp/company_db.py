from __future__ import annotations

from app.services.mcp.base import BaseMCPService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CompanyDBMCPService(BaseMCPService):
    service_name: str = "company_db"

    async def query(self, sql: str, params: dict) -> list[dict]:
        if self._use_real_mode():
            # TODO(M2): 실제 CompanyDB MCP SDK 호출 구현
            raise NotImplementedError("CompanyDB MCP 실제 연동 미구현 — MCP_COMPANY_DB_REAL 해제 후 사용")
        logger.info("CompanyDBMCPService(mock) query sql=%s", sql)
        return []
