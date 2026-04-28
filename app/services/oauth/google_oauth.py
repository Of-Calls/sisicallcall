"""
Google OAuth 2.0 구현 — Gmail / Google Calendar 공통 기반.

env:
  GOOGLE_OAUTH_CLIENT_ID       (필수)
  GOOGLE_OAUTH_CLIENT_SECRET   (필수)
  GOOGLE_OAUTH_REDIRECT_URI    (필수)

scope 예:
  Gmail:    https://www.googleapis.com/auth/gmail.send
  Calendar: https://www.googleapis.com/auth/calendar
"""
from __future__ import annotations

import os
import urllib.parse

import httpx

from app.services.oauth.base import BaseOAuthProvider, TokenResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


class _GoogleOAuthBase(BaseOAuthProvider):
    _default_scopes: list[str] = []

    def _client_id(self) -> str:
        return os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")

    def _client_secret(self) -> str:
        return os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "")

    def _redirect_uri(self) -> str:
        return os.getenv("GOOGLE_OAUTH_REDIRECT_URI", "")

    def get_authorize_url(self, state: str, scopes: list[str] | None = None) -> str:
        chosen = scopes or self._default_scopes
        params = {
            "client_id": self._client_id(),
            "redirect_uri": self._redirect_uri(),
            "response_type": "code",
            "scope": " ".join(chosen),
            "access_type": "offline",
            "prompt": "consent",
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
                "grant_type": "authorization_code",
            })
            resp.raise_for_status()
            data = resp.json()

        email, account_id = await self._fetch_userinfo(data["access_token"])
        return TokenResult(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            expires_in=data.get("expires_in"),
            scope=data.get("scope", ""),
            token_type=data.get("token_type", "Bearer"),
            external_account_id=account_id,
            external_account_email=email,
            raw=data,
        )

    async def refresh_token(self, refresh_token: str) -> TokenResult:
        async with httpx.AsyncClient() as client:
            resp = await client.post(_TOKEN_URL, data={
                "refresh_token": refresh_token,
                "client_id": self._client_id(),
                "client_secret": self._client_secret(),
                "grant_type": "refresh_token",
            })
            resp.raise_for_status()
            data = resp.json()

        return TokenResult(
            access_token=data["access_token"],
            refresh_token=refresh_token,
            expires_in=data.get("expires_in"),
            scope=data.get("scope", ""),
            token_type=data.get("token_type", "Bearer"),
            raw=data,
        )

    async def _fetch_userinfo(self, access_token: str) -> tuple[str, str]:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    _USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                resp.raise_for_status()
                info = resp.json()
            return info.get("email", ""), info.get("id", "")
        except Exception as exc:
            logger.warning("Google userinfo 조회 실패: %s", exc)
            return "", ""


class GoogleGmailOAuth(_GoogleOAuthBase):
    provider_name = "google_gmail"
    _default_scopes = [
        "https://www.googleapis.com/auth/gmail.send",
        "openid",
        "email",
    ]


class GoogleCalendarOAuth(_GoogleOAuthBase):
    provider_name = "google_calendar"
    _default_scopes = [
        "https://www.googleapis.com/auth/calendar",
        "openid",
        "email",
    ]
