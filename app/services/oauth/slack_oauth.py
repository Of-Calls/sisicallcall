"""
Slack OAuth 2.0 구현.

env:
  SLACK_CLIENT_ID        (필수)
  SLACK_CLIENT_SECRET    (필수)
  SLACK_REDIRECT_URI     (필수)

scope 예: chat:write, channels:read, users:read
"""
from __future__ import annotations

import os
import urllib.parse

import httpx

from app.services.oauth.base import BaseOAuthProvider, TokenResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

_AUTH_URL = "https://slack.com/oauth/v2/authorize"
_TOKEN_URL = "https://slack.com/api/oauth.v2.access"


class SlackOAuth(BaseOAuthProvider):
    provider_name = "slack"

    _default_scopes = ["chat:write", "channels:read", "users:read"]

    def _client_id(self) -> str:
        return os.getenv("SLACK_CLIENT_ID", "")

    def _client_secret(self) -> str:
        return os.getenv("SLACK_CLIENT_SECRET", "")

    def _redirect_uri(self) -> str:
        return os.getenv("SLACK_REDIRECT_URI", "")

    def get_authorize_url(self, state: str, scopes: list[str] | None = None) -> str:
        chosen = scopes or self._default_scopes
        params = {
            "client_id": self._client_id(),
            "redirect_uri": self._redirect_uri(),
            "scope": ",".join(chosen),
            "state": state,
        }
        return f"{_AUTH_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> TokenResult:
        async with httpx.AsyncClient() as client:
            resp = await client.post(_TOKEN_URL, data={
                "code": code,
                "client_id": self._client_id(),
                "client_secret": self._client_secret(),
                "redirect_uri": redirect_uri,
            })
            resp.raise_for_status()
            data = resp.json()

        if not data.get("ok"):
            raise RuntimeError(f"Slack OAuth 실패: {data.get('error', 'unknown')}")

        authed = data.get("authed_user", {})
        team = data.get("team", {})
        bot = data.get("bot_user_id", "")

        return TokenResult(
            access_token=data.get("access_token", authed.get("access_token", "")),
            refresh_token=None,
            scope=data.get("scope", authed.get("scope", "")),
            token_type="Bearer",
            external_account_id=authed.get("id") or bot,
            external_workspace_id=team.get("id", ""),
            external_workspace_name=team.get("name", ""),
            raw=data,
        )

    async def refresh_token(self, refresh_token: str) -> TokenResult:
        raise NotImplementedError("Slack Bot 토큰은 만료되지 않으므로 refresh 불필요")
