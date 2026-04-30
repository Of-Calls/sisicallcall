"""
BaseOAuthProvider — 모든 OAuth 프로바이더의 추상 기반 클래스.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TokenResult:
    access_token: str
    refresh_token: str | None = None
    expires_in: int | None = None
    scope: str = ""
    token_type: str = "Bearer"
    external_account_id: str | None = None
    external_account_email: str | None = None
    external_workspace_id: str | None = None
    external_workspace_name: str | None = None
    raw: dict | None = None


class BaseOAuthProvider(ABC):
    provider_name: str = ""

    @abstractmethod
    def get_authorize_url(self, state: str, scopes: list[str] | None = None) -> str:
        """인가 URL 생성 — 브라우저 리다이렉트용."""

    @abstractmethod
    async def exchange_code(self, code: str, redirect_uri: str) -> TokenResult:
        """authorization code → access/refresh token 교환."""

    @abstractmethod
    async def refresh_token(self, refresh_token: str) -> TokenResult:
        """refresh token → 새 access token 발급."""
