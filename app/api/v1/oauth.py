"""
OAuth 2.0 연동 API — SaaS 고객사가 자신의 서비스를 연결하는 엔드포인트.

등록 (app/main.py):
    app.include_router(oauth_router, prefix="/api/v1/oauth", tags=["oauth"])

흐름:
  1. GET /api/v1/oauth/{provider}/authorize?tenant_id=&return_url=
     → 프로바이더 OAuth URL로 리다이렉트

  2. GET /api/v1/oauth/{provider}/callback?code=&state=
     → code 교환, 토큰 암호화 저장, JSON 반환

지원 provider: google_gmail, google_calendar, slack, jira
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from app.models.tenant_integration import IntegrationStatus, TenantIntegration
from app.repositories.tenant_integration_repo import upsert_integration
from app.services.oauth.state import create_oauth_state, verify_oauth_state
from app.services.oauth.token_crypto import encrypt_token
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# ── 프로바이더 레지스트리 ─────────────────────────────────────────────────────

def _get_provider(name: str):
    from app.services.oauth.google_oauth import GoogleGmailOAuth, GoogleCalendarOAuth
    from app.services.oauth.slack_oauth import SlackOAuth
    from app.services.oauth.jira_oauth import JiraOAuth

    registry = {
        "google_gmail": GoogleGmailOAuth(),
        "google_calendar": GoogleCalendarOAuth(),
        "slack": SlackOAuth(),
        "jira": JiraOAuth(),
    }
    provider = registry.get(name)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"지원하지 않는 provider: {name}")
    return provider


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@router.get("/{provider}/authorize")
async def authorize(
    provider: str,
    tenant_id: str = Query(..., description="연동할 테넌트 ID"),
    return_url: str = Query("", description="연동 완료 후 리다이렉트 URL"),
    scopes: str = Query("", description="요청 scope (쉼표 구분, 선택)"),
):
    """프로바이더 OAuth 인가 URL로 리다이렉트한다."""
    oauth = _get_provider(provider)
    state = create_oauth_state(tenant_id=tenant_id, provider=provider, return_url=return_url)
    scope_list = [s.strip() for s in scopes.split(",") if s.strip()] or None
    url = oauth.get_authorize_url(state=state, scopes=scope_list)
    return RedirectResponse(url=url, status_code=302)


@router.get("/{provider}/callback")
async def callback(
    provider: str,
    code: str = Query(...),
    state: str = Query(...),
):
    """OAuth callback — code 교환, 토큰 암호화 저장."""
    entry = verify_oauth_state(state)
    if entry is None:
        raise HTTPException(status_code=400, detail="OAuth state 검증 실패 또는 만료")
    if entry.provider != provider:
        raise HTTPException(status_code=400, detail="provider 불일치")

    oauth = _get_provider(provider)
    redirect_uri = _redirect_uri_for(provider)

    try:
        token_result = await oauth.exchange_code(code=code, redirect_uri=redirect_uri)
    except Exception as exc:
        logger.error("OAuth code exchange 실패 provider=%s err=%s", provider, exc)
        raise HTTPException(status_code=502, detail=f"토큰 교환 실패: {exc}") from exc

    access_enc = encrypt_token(token_result.access_token)
    refresh_enc = encrypt_token(token_result.refresh_token) if token_result.refresh_token else None

    expires_at: datetime | None = None
    if token_result.expires_in:
        expires_at = datetime.utcnow() + timedelta(seconds=token_result.expires_in)

    scopes_list = [s for s in (token_result.scope or "").split() if s]

    integration = TenantIntegration(
        tenant_id=entry.tenant_id,
        provider=provider,
        status=IntegrationStatus.connected,
        scopes=scopes_list,
        access_token_encrypted=access_enc,
        refresh_token_encrypted=refresh_enc,
        token_type=token_result.token_type,
        expires_at=expires_at,
        external_account_id=token_result.external_account_id,
        external_account_email=token_result.external_account_email,
        external_workspace_id=token_result.external_workspace_id,
        external_workspace_name=token_result.external_workspace_name,
    )
    upsert_integration(integration)

    logger.info(
        "OAuth 연동 완료 tenant_id=%s provider=%s account=%s",
        entry.tenant_id, provider, token_result.external_account_email,
    )

    return {
        "status": "connected",
        "tenant_id": entry.tenant_id,
        "provider": provider,
        "account_email": token_result.external_account_email,
        "workspace_name": token_result.external_workspace_name,
        "expires_at": expires_at.isoformat() if expires_at else None,
        "return_url": entry.return_url,
    }


@router.get("/{provider}/status")
async def get_status(
    provider: str,
    tenant_id: str = Query(...),
):
    """특정 테넌트의 연동 상태 조회."""
    from app.repositories.tenant_integration_repo import get_integration

    integration = get_integration(tenant_id, provider)
    if integration is None:
        return {"status": "not_connected", "tenant_id": tenant_id, "provider": provider}

    return {
        "status": integration.status.value,
        "tenant_id": tenant_id,
        "provider": provider,
        "account_email": integration.external_account_email,
        "workspace_name": integration.external_workspace_name,
        "expires_at": integration.expires_at.isoformat() if integration.expires_at else None,
        "scopes": integration.scopes,
    }


@router.delete("/{provider}/disconnect")
async def disconnect(
    provider: str,
    tenant_id: str = Query(...),
):
    """연동 해제."""
    from app.repositories.tenant_integration_repo import mark_disconnected

    ok = mark_disconnected(tenant_id, provider)
    if not ok:
        raise HTTPException(status_code=404, detail="연동 정보 없음")
    return {"status": "disconnected", "tenant_id": tenant_id, "provider": provider}


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _redirect_uri_for(provider: str) -> str:
    """provider별 콜백 URI 결정.

    authorize URL 생성(get_authorize_url)과 token exchange(exchange_code) 양쪽이
    동일한 redirect_uri를 사용해야 Google이 요청을 수락한다.
    GoogleGmailOAuth._redirect_uri() / GoogleCalendarOAuth._redirect_uri()와
    동일한 우선순위 로직을 따른다.
    """
    if provider == "google_calendar":
        return (
            os.getenv("GOOGLE_CALENDAR_REDIRECT_URI")
            or os.getenv("GOOGLE_OAUTH_REDIRECT_URI", "")
        )
    if provider == "google_gmail":
        return (
            os.getenv("GOOGLE_GMAIL_REDIRECT_URI")
            or os.getenv("GOOGLE_OAUTH_REDIRECT_URI", "")
        )
    if provider == "slack":
        return os.getenv("SLACK_REDIRECT_URI", "")
    if provider == "jira":
        return os.getenv("ATLASSIAN_REDIRECT_URI", "")
    return ""
