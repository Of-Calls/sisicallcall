"""
BaseMCPConnector — MCP Connector 계층 공통 기반.

각 Connector는 이 클래스를 상속하고 execute()를 구현한다.
import 시점에 외부 API / MCP 서버 연결을 만들지 않는다.

── 표준 반환 형식 ────────────────────────────────────────────────────────────
{
  "status":      "success" | "failed" | "skipped",
  "external_id": "...",          # 외부 시스템이 발급한 ID (없으면 None)
  "result":      {...},          # connector별 상세 결과
  "error":       None | "..."    # 오류 메시지 (success면 None)
}

── real mode 분기 원칙 ──────────────────────────────────────────────────────
- is_real_mode() == False  → mock result (status=success)
- is_real_mode() == True, validate_config() 실패  → skipped + error
- is_real_mode() == True, validate_config() 성공  → 실제 통합 (미구현 시 skipped)
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from app.utils.logger import get_logger

logger = get_logger(__name__)


class BaseMCPConnector(ABC):
    """모든 MCP Connector의 공통 추상 기반 클래스."""

    connector_name: str = "base"
    _real_mode_env: str = ""           # 예: "GMAIL_MCP_REAL"
    _required_config: tuple[str, ...] = ()  # real mode 필수 env var

    # ── mode 판단 ─────────────────────────────────────────────────────────────

    def is_real_mode(self) -> bool:
        """환경변수 {_real_mode_env}=true|1 이면 True."""
        if not self._real_mode_env:
            return False
        return os.getenv(self._real_mode_env, "").lower() in ("1", "true")

    def validate_config(self) -> tuple[bool, str | None]:
        """real mode에 필요한 env var이 모두 설정되었는지 확인한다.

        Returns:
            (True, None)        → 설정 완료
            (False, error_msg)  → 누락 env var 목록 포함 메시지
        """
        missing = [k for k in self._required_config if not os.getenv(k)]
        if missing:
            return False, f"missing env: {', '.join(missing)}"
        return True, None

    # ── 추상 메서드 ───────────────────────────────────────────────────────────

    @abstractmethod
    async def execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        """action_type에 따라 외부 도구를 호출하고 표준 결과를 반환한다."""

    # ── 결과 헬퍼 ─────────────────────────────────────────────────────────────

    def _success(self, external_id: str | None, result: dict) -> dict:
        return {
            "status": "success",
            "external_id": external_id,
            "result": result,
            "error": None,
        }

    def _skipped(self, error: str, result: dict | None = None) -> dict:
        return {
            "status": "skipped",
            "external_id": None,
            "result": result or {},
            "error": error,
        }

    def _failed(self, error: str, result: dict | None = None) -> dict:
        return {
            "status": "failed",
            "external_id": None,
            "result": result or {},
            "error": error,
        }
