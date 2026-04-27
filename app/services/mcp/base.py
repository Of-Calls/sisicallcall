"""
MCP Service Base Class.

새 MCP service를 추가하는 방법:
1. app/services/mcp/{name}.py 파일 생성
2. BaseMCPService 를 상속하고 service_name = "{name}" 설정
3. 메서드 안에서 _use_real_mode() 로 분기:
     if self._use_real_mode():
         # 실제 외부 SDK 호출 (NotImplementedError 허용 → executor가 failed로 처리)
     # mock 구현 (항상 동작)
4. 대응하는 action handler 에서 이 서비스를 사용

환경 변수 규칙:
  MCP_{SERVICE_NAME_UPPER}_REAL=1  →  real mode 활성화
  예: MCP_GMAIL_REAL=1, MCP_COMPANY_DB_REAL=1, MCP_CALENDAR_REAL=1

import 시점에 외부 SDK를 로드하지 않으므로 credential 없이도 안전하게 import 된다.
"""
from __future__ import annotations

import os
from abc import ABC


class BaseMCPService(ABC):
    """MCP 서비스 공통 기반 — 모든 MCP 클라이언트가 상속."""

    service_name: str = "base"

    @classmethod
    def _use_real_mode(cls) -> bool:
        """환경 변수 MCP_{SERVICE_NAME}_REAL=1|true 이면 True."""
        env_key = f"MCP_{cls.service_name.upper()}_REAL"
        return os.getenv(env_key, "").lower() in ("1", "true")

    @classmethod
    def _is_mock_mode(cls) -> bool:
        """_use_real_mode() 의 반전 — mock branch 가독성용."""
        return not cls._use_real_mode()
