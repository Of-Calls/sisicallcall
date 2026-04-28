"""
Company DB MCP Connector.

지원 action_type:
  - create_voc_issue
  - add_priority_queue

── real mode env ─────────────────────────────────────────────────────────────
  COMPANY_DB_MCP_REAL=true   OR  MCP_COMPANY_DB_REAL=true  (둘 중 하나)
  COMPANY_DB_BASE_URL        API 서버 URL (선택)
  COMPANY_DB_MCP_SERVER_URL  MCP 서버 URL (선택)

── mock mode ─────────────────────────────────────────────────────────────────
  status: success
  external_id: VOC-MOCK-{call_id}
  result: {created, issue_id, tier, priority, primary_category, reason, summary_short, mock}

── real mode 설정 부족 ────────────────────────────────────────────────────────
  status: skipped
  error: "company_db_connector_not_configured"
"""
from __future__ import annotations

import os

from app.services.mcp.connectors.base import BaseMCPConnector
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CompanyDBConnector(BaseMCPConnector):
    connector_name = "company_db"
    # 두 가지 env var 중 하나라도 true이면 real mode
    _real_mode_env = "COMPANY_DB_MCP_REAL"

    def is_real_mode(self) -> bool:
        """COMPANY_DB_MCP_REAL 또는 MCP_COMPANY_DB_REAL 중 하나가 true이면 real mode."""
        v1 = os.getenv("COMPANY_DB_MCP_REAL", "").lower()
        v2 = os.getenv("MCP_COMPANY_DB_REAL", "").lower()
        return v1 in ("1", "true") or v2 in ("1", "true")

    def validate_config(self) -> tuple[bool, str | None]:
        # 필수 env 없음 — URL은 선택이므로 항상 통과
        return True, None

    async def execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        logger.info(
            "CompanyDBConnector call_id=%s action_type=%s real_mode=%s",
            call_id, action_type, self.is_real_mode(),
        )

        if not self.is_real_mode():
            return self._mock(params, call_id)

        ok, err = self.validate_config()
        if not ok:
            logger.warning("CompanyDBConnector: config 부족 call_id=%s err=%s", call_id, err)
            return self._skipped("company_db_connector_not_configured")

        return await self._execute_real(action_type, params, call_id=call_id)

    def _mock(self, params: dict, call_id: str) -> dict:
        issue_id = f"VOC-MOCK-{call_id}"
        return self._success(
            external_id=issue_id,
            result={
                "created": True,
                "issue_id": issue_id,
                "tier": params.get("tier", "medium"),
                "priority": params.get("priority", "medium"),
                "primary_category": params.get("primary_category", ""),
                "reason": params.get("reason", ""),
                "summary_short": params.get("summary_short", ""),
                "mock": True,
            },
        )

    async def _execute_real(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
    ) -> dict:
        # TODO: 실제 Company DB API / MCP 서버 연동 구현
        # base_url = os.getenv("COMPANY_DB_BASE_URL")
        # server_url = os.getenv("COMPANY_DB_MCP_SERVER_URL")
        logger.warning(
            "CompanyDBConnector: real mode TODO — skipped call_id=%s action_type=%s",
            call_id, action_type,
        )
        return self._skipped("company_db_mcp_real_not_implemented")
