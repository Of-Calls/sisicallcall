"""
Google OAuth 2.0 구현 — Gmail / Google Calendar 공통 기반.

env:
  GOOGLE_OAUTH_CLIENT_ID       (필수)
  GOOGLE_OAUTH_CLIENT_SECRET   (필수)

provider별 redirect URI (우선순위):
  google_calendar → GOOGLE_CALENDAR_REDIRECT_URI → GOOGLE_OAUTH_REDIRECT_URI
  google_gmail    → GOOGLE_GMAIL_REDIRECT_URI    → GOOGLE_OAUTH_REDIRECT_URI

authorize URL 생성과 token exchange 양쪽에 동일한 redirect_uri가 사용되도록
_redirect_uri() 메서드를 통해 결정한다.

Google Cloud Console 승인된 리디렉션 URI 등록 필요:
  http://localhost:8000/api/v1/oauth/google_calendar/callback
  http://localhost:8000/api/v1/oauth/google_gmail/callback
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
    _redirect_uri_env: str = ""  # 서브클래스가 provider별 env 이름을 지정

    def _client_id(self) -> str:
        return os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")

    def _client_secret(self) -> str:
        return os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "")

    def _redirect_uri(self) -> str:
        """provider별 redirect URI를 반환한다.

        우선순위:
          1. _redirect_uri_env 에 지정된 env var (예: GOOGLE_CALENDAR_REDIRECT_URI)
          2. 하위 호환 GOOGLE_OAUTH_REDIRECT_URI
        """
        if self._redirect_uri_env:
            val = os.getenv(self._redirect_uri_env, "")
            if val:
                return val
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
        """redirect_uri는 authorize URL 생성 시 사용한 값과 동일해야 한다.

        caller(oauth.py callback 엔드포인트)는 _redirect_uri()와 같은 값을
        넘겨야 한다. _redirect_uri_for() 헬퍼가 이를 보장한다.
        """
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
    _redirect_uri_env = "GOOGLE_GMAIL_REDIRECT_URI"
    _default_scopes = [
        "https://www.googleapis.com/auth/gmail.send",
        "openid",
        "email",
    ]


class GoogleCalendarOAuth(_GoogleOAuthBase):
    provider_name = "google_calendar"
    _redirect_uri_env = "GOOGLE_CALENDAR_REDIRECT_URI"
    _default_scopes = [
        "https://www.googleapis.com/auth/calendar",
        "openid",
        "email",
    ]
