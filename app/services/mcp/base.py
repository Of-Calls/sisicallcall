from __future__ import annotations

import os
from abc import ABC


class BaseMCPService(ABC):
    """MCP 서비스 공통 기반 — 모든 MCP 클라이언트가 상속."""

    service_name: str = "base"

    @classmethod
    def _use_real_mode(cls) -> bool:
        """환경 변수 MCP_{SERVICE_NAME}_REAL=1|true 이면 실제 모드."""
        env_key = f"MCP_{cls.service_name.upper()}_REAL"
        return os.getenv(env_key, "").lower() in ("1", "true")
