from __future__ import annotations

import os

from app.utils.logger import get_logger

logger = get_logger(__name__)


class CompanyDBAction:
    async def execute(self, action: dict, *, call_id: str, tenant_id: str = "") -> dict:
        params = action.get("params", {})
        logger.info("CompanyDBAction call_id=%s action_type=%s", call_id, action.get("action_type"))

        if os.getenv("MCP_COMPANY_DB_REAL", "").lower() in ("1", "true"):
            return await self._execute_real(params, call_id=call_id)

        issue_id = f"VOC-MOCK-{call_id}"
        return {
            "external_id": issue_id,
            "status": "success",
            "result": {
                "created": True,
                "issue_id": issue_id,
                "tier": params.get("tier", "medium"),
                "priority": params.get("priority", "medium"),
                "primary_category": params.get("primary_category", ""),
                "reason": params.get("reason", ""),
                "summary_short": params.get("summary_short", ""),
                "mock": True,
            },
        }

    async def _execute_real(self, params: dict, *, call_id: str) -> dict:
        raise NotImplementedError("CompanyDB 실제 연동은 MCP 구성 후 구현")
