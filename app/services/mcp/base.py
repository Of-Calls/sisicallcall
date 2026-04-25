from __future__ import annotations
from abc import ABC


class BaseMCPService(ABC):
    """MCP 서비스 공통 기반 — 모든 MCP 클라이언트가 상속."""

    service_name: str = "base"
