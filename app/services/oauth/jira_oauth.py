"""
Atlassian Jira OAuth 2.0 (3LO) 구현.

env:
  ATLASSIAN_CLIENT_ID        (필수)
  ATLASSIAN_CLIENT_SECRET    (필수)
  ATLASSIAN_REDIRECT_URI     (필수)

scope 예: read:jira-work write:jira-work read:jira-user offline_access
"""
from __future__ import annotations

import os
import urllib.parse

import httpx

from app.services.oauth.base import BaseOAuthProvider, TokenResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

_AUTH_URL = "https://auth.atlassian.com/authorize"
_TOKEN_URL = "https://auth.atlassian.com/oauth/token"
_RESOURCES_URL = "https://api.atlassian.com/oauth/token/accessible-resources"


class JiraOAuth(BaseOAuthProvider):
    provider_name = "jira"

    _default_scopes = [
        "read:jira-work",
        "write:jira-work",
        "read:jira-user",
        "offline_access",
    ]

    def _client_id(self) -> str:
        return os.getenv("ATLASSIAN_CLIENT_ID", "")

    def _client_secret(self) -> str:
        return os.getenv("ATLASSIAN_CLIENT_SECRET", "")

    def _redirect_uri(self) -> str:
        return os.getenv("ATLASSIAN_REDIRECT_URI", "")

    def get_authorize_url(self, state: str, scopes: list[str] | None = None) -> str:
        chosen = scopes or self._default_scopes
        params = {
            "audience": "api.atlassian.com",
            "client_id": self._client_id(),
            "redirect_uri": self._redirect_uri(),
            "response_type": "code",
            "scope": " ".join(chosen),
            "state": state,
            "prompt": "consent",
        }
        return f"{_AUTH_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> TokenResult:
        async with httpx.AsyncClient() as client:
            resp = await client.post(_TOKEN_URL, json={
                "grant_type": "authorization_code",
                "client_id": self._client_id(),
                "client_secret": self._client_secret(),
                "code": code,
                "redirect_uri": redirect_uri,
            })
            resp.raise_for_status()
            data = resp.json()

        workspace_id, workspace_name = await self._fetch_workspace(data["access_token"])

        return TokenResult(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            expires_in=data.get("expires_in"),
            scope=data.get("scope", ""),
            token_type=data.get("token_type", "Bearer"),
            external_workspace_id=workspace_id,
            external_workspace_name=workspace_name,
            raw=data,
        )

    async def refresh_token(self, refresh_token: str) -> TokenResult:
        async with httpx.AsyncClient() as client:
            resp = await client.post(_TOKEN_URL, json={
                "grant_type": "refresh_token",
                "client_id": self._client_id(),
                "client_secret": self._client_secret(),
                "refresh_token": refresh_token,
            })
            resp.raise_for_status()
            data = resp.json()

        return TokenResult(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", refresh_token),
            expires_in=data.get("expires_in"),
            scope=data.get("scope", ""),
            token_type=data.get("token_type", "Bearer"),
            raw=data,
        )

    async def _fetch_workspace(self, access_token: str) -> tuple[str, str]:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    _RESOURCES_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                resp.raise_for_status()
                resources = resp.json()
            if resources:
                first = resources[0]
                return first.get("id", ""), first.get("name", "")
        except Exception as exc:
            logger.warning("Jira accessible-resources 조회 실패: %s", exc)
        return "", ""
