"""
OAuth API 라우터 테스트.

[참고] app.main은 NeMo(call.py) 의존성으로 인해 테스트 환경에서 직접 import 불가.
       oauth_client fixture: /oauth prefix 최소 앱 (기능 검증)
       oauth_full_client fixture: /api/v1/oauth prefix 최소 앱 (main.py 등록 경로 검증)
       실제 main.py 등록: app.include_router(oauth_router, prefix="/api/v1/oauth", tags=["oauth"])

검증 범위:
  1.  GET /oauth/{provider}/authorize → 302 리다이렉트, state 포함
  2.  지원하지 않는 provider → 404
  3.  GET /oauth/{provider}/status — 미연동 시 not_connected
  4.  GET /oauth/{provider}/status — 연동 후 connected
  5.  DELETE /oauth/{provider}/disconnect — 연동 해제
  6.  DELETE /oauth/{provider}/disconnect — 없는 연동 → 404
  7.  GET /oauth/{provider}/callback — state 만료/무효 → 400
  8.  GET /oauth/{provider}/callback — provider 불일치 → 400
  9.  authorize 후 state가 verify_oauth_state로 검증 가능
  10. authorize에 scopes 파라미터 전달 시 URL에 반영
  11. oauth 라우터가 /api/v1/oauth prefix로 마운트됐을 때 동일하게 동작 (main.py 경로 검증)
  12. oauth 라우터 routes 목록에 authorize/callback/status/disconnect 경로 포함
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ── FastAPI app fixture (oauth 라우터만 포함한 최소 앱) ────────────────────────

@pytest.fixture(scope="module")
def oauth_client():
    from fastapi import FastAPI
    from app.api.v1.oauth import router as oauth_router

    app = FastAPI()
    app.include_router(oauth_router, prefix="/oauth")
    return TestClient(app, follow_redirects=False)


@pytest.fixture(autouse=True)
def _clear():
    """테스트마다 OAuth state와 통합 저장소를 격리한다.

    실서비스 .env 가 db mode 일 수 있어 테스트 단에서는 메모리 모드 인스턴스로
    싱글턴을 교체한다 — Postgres 가 없거나 UUID 가 아닌 tenant_id 를 쓰는
    테스트 케이스에서도 동일하게 동작하도록 한다.
    """
    from app.services.oauth import state as state_mod
    from app.repositories import tenant_integration_repo as repo_mod

    state_mod.clear_oauth_states()
    original_repo = repo_mod.tenant_integration_repo
    repo_mod.tenant_integration_repo = repo_mod.TenantIntegrationRepository(storage="memory")
    try:
        yield
    finally:
        state_mod.clear_oauth_states()
        repo_mod.tenant_integration_repo = original_repo


# ── 1. authorize → 302 리다이렉트 + state 포함 ───────────────────────────────

def test_authorize_redirects_to_provider(oauth_client, monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-cid")
    monkeypatch.setenv("GOOGLE_OAUTH_REDIRECT_URI", "https://app.example.com/cb")

    resp = oauth_client.get(
        "/oauth/google_gmail/authorize",
        params={"tenant_id": "tenant-a", "return_url": "https://app.example.com"},
    )

    assert resp.status_code == 302
    location = resp.headers["location"]
    assert "accounts.google.com" in location
    assert "state=" in location


# ── 2. 지원하지 않는 provider → 404 ─────────────────────────────────────────

def test_authorize_unknown_provider_returns_404(oauth_client):
    resp = oauth_client.get(
        "/oauth/unknown_provider/authorize",
        params={"tenant_id": "tenant-a"},
    )
    assert resp.status_code == 404


# ── 3. status — 미연동 → not_connected ───────────────────────────────────────

def test_status_not_connected(oauth_client):
    resp = oauth_client.get(
        "/oauth/google_gmail/status",
        params={"tenant_id": "tenant-b"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "not_connected"
    assert data["tenant_id"] == "tenant-b"
    assert data["provider"] == "google_gmail"


# ── 4. status — 연동 후 connected ────────────────────────────────────────────

def test_status_connected_after_upsert(oauth_client):
    from app.models.tenant_integration import TenantIntegration, IntegrationStatus
    from app.repositories.tenant_integration_repo import upsert_integration

    upsert_integration(TenantIntegration(
        tenant_id="tenant-c",
        provider="slack",
        status=IntegrationStatus.connected,
        external_account_email="bot@workspace.slack.com",
        external_workspace_name="My Workspace",
    ))

    resp = oauth_client.get(
        "/oauth/slack/status",
        params={"tenant_id": "tenant-c"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "connected"
    assert data["account_email"] == "bot@workspace.slack.com"
    assert data["workspace_name"] == "My Workspace"


# ── 5. disconnect — 연동 해제 ─────────────────────────────────────────────────

def test_disconnect_success(oauth_client):
    from app.models.tenant_integration import TenantIntegration
    from app.repositories.tenant_integration_repo import upsert_integration, get_integration
    from app.models.tenant_integration import IntegrationStatus

    upsert_integration(TenantIntegration(tenant_id="tenant-d", provider="jira"))

    resp = oauth_client.delete(
        "/oauth/jira/disconnect",
        params={"tenant_id": "tenant-d"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "disconnected"

    result = get_integration("tenant-d", "jira")
    assert result is not None
    assert result.status == IntegrationStatus.disconnected


# ── 6. disconnect — 없는 연동 → 404 ──────────────────────────────────────────

def test_disconnect_nonexistent_returns_404(oauth_client):
    resp = oauth_client.delete(
        "/oauth/jira/disconnect",
        params={"tenant_id": "no-such-tenant"},
    )
    assert resp.status_code == 404


# ── 7. callback — state 무효 → 400 ───────────────────────────────────────────

def test_callback_invalid_state_returns_400(oauth_client):
    resp = oauth_client.get(
        "/oauth/google_gmail/callback",
        params={"code": "some-code", "state": "invalid-state-xyz"},
    )
    assert resp.status_code == 400
    assert "state" in resp.json()["detail"].lower()


# ── 8. callback — provider 불일치 → 400 ──────────────────────────────────────

def test_callback_provider_mismatch_returns_400(oauth_client):
    from app.services.oauth.state import create_oauth_state

    # state를 jira provider로 생성했지만, google_gmail callback으로 요청
    state = create_oauth_state("tenant-e", "jira", "")

    resp = oauth_client.get(
        "/oauth/google_gmail/callback",
        params={"code": "some-code", "state": state},
    )
    assert resp.status_code == 400
    assert "provider" in resp.json()["detail"]


# ── 9. authorize 후 state가 verify_oauth_state로 검증 가능 ───────────────────

def test_authorize_creates_verifiable_state(oauth_client, monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-cid")
    monkeypatch.setenv("GOOGLE_OAUTH_REDIRECT_URI", "https://app.example.com/cb")

    from app.services.oauth.state import _state_store

    before_count = len(_state_store)
    resp = oauth_client.get(
        "/oauth/google_gmail/authorize",
        params={"tenant_id": "tenant-f"},
    )
    assert resp.status_code == 302
    # state가 store에 추가됐는지 확인
    assert len(_state_store) == before_count + 1


# ── 10. authorize scopes 파라미터 반영 ───────────────────────────────────────

def test_authorize_with_custom_scopes(oauth_client, monkeypatch):
    monkeypatch.setenv("SLACK_CLIENT_ID", "slack-cid")
    monkeypatch.setenv("SLACK_REDIRECT_URI", "https://app.example.com/slack/cb")

    resp = oauth_client.get(
        "/oauth/slack/authorize",
        params={
            "tenant_id": "tenant-g",
            "scopes": "chat:write,channels:read",
        },
    )
    assert resp.status_code == 302
    location = resp.headers["location"]
    assert "chat" in location


# ── main.py 등록 경로 검증 (/api/v1/oauth prefix) ─────────────────────────────

@pytest.fixture(scope="module")
def oauth_full_client():
    """/api/v1/oauth prefix — main.py와 동일한 마운트 경로로 검증."""
    from fastapi import FastAPI
    from app.api.v1.oauth import router as oauth_router

    app = FastAPI()
    app.include_router(oauth_router, prefix="/api/v1/oauth", tags=["oauth"])
    return TestClient(app, follow_redirects=False)


# ── 11. /api/v1/oauth prefix 경로에서 동일하게 동작 ──────────────────────────

def test_full_prefix_authorize_redirects(oauth_full_client, monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-cid")
    monkeypatch.setenv("GOOGLE_OAUTH_REDIRECT_URI", "https://app.example.com/cb")

    resp = oauth_full_client.get(
        "/api/v1/oauth/google_gmail/authorize",
        params={"tenant_id": "tenant-x"},
    )
    assert resp.status_code == 302
    assert "accounts.google.com" in resp.headers["location"]


def test_full_prefix_status_not_connected(oauth_full_client):
    resp = oauth_full_client.get(
        "/api/v1/oauth/google_gmail/status",
        params={"tenant_id": "tenant-x"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "not_connected"


# ── 12. router routes 목록에 필수 경로 포함 ──────────────────────────────────

def test_oauth_router_has_required_routes():
    from app.api.v1.oauth import router as oauth_router

    paths = {route.path for route in oauth_router.routes}
    assert "/{provider}/authorize"   in paths
    assert "/{provider}/callback"    in paths
    assert "/{provider}/status"      in paths
    assert "/{provider}/disconnect"  in paths


# ── 13. callback — return_url 없으면 기존 JSON 응답 유지 ─────────────────────

def _stub_exchange_code(monkeypatch, *, scope: str = "openid email", workspace_id: str | None = None,
                        workspace_name: str | None = None, raw: dict | None = None):
    """exchange_code를 외부 HTTP 호출 없이 고정 TokenResult로 stub."""
    from app.services.oauth.base import TokenResult
    from app.services.oauth import google_oauth, slack_oauth, jira_oauth

    async def _fake(self, code: str, redirect_uri: str) -> TokenResult:
        return TokenResult(
            access_token="fake-access-token",
            refresh_token="fake-refresh-token",
            expires_in=3600,
            scope=scope,
            external_account_id="acc-1",
            external_account_email="user@example.com",
            external_workspace_id=workspace_id,
            external_workspace_name=workspace_name,
            raw=raw or {},
        )

    monkeypatch.setattr(google_oauth._GoogleOAuthBase, "exchange_code", _fake)
    monkeypatch.setattr(slack_oauth.SlackOAuth, "exchange_code", _fake)
    monkeypatch.setattr(jira_oauth.JiraOAuth, "exchange_code", _fake)


def test_callback_without_return_url_returns_json(oauth_client, monkeypatch):
    """return_url 없으면 기존 JSON 응답 유지 (API 직접 호출 호환)."""
    from app.services.oauth.state import create_oauth_state

    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", "Cw3aE_Yp8jU2kZjKmCQ41eFPwJfH-tYxKXG2QJAcIIE=")
    _stub_exchange_code(monkeypatch)

    state = create_oauth_state("tenant-cb1", "google_calendar", return_url="")

    resp = oauth_client.get(
        "/oauth/google_calendar/callback",
        params={"code": "fake-code", "state": state},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "connected"
    assert data["provider"] == "google_calendar"
    assert data["account_email"] == "user@example.com"


def test_callback_with_safe_return_url_redirects(oauth_client, monkeypatch):
    """return_url이 허용된 frontend origin이면 RedirectResponse(302)로 응답."""
    from app.services.oauth.state import create_oauth_state

    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", "Cw3aE_Yp8jU2kZjKmCQ41eFPwJfH-tYxKXG2QJAcIIE=")
    _stub_exchange_code(monkeypatch)

    state = create_oauth_state(
        "tenant-cb2",
        "google_gmail",
        return_url="http://localhost:5173/dashboard/integrations",
    )

    resp = oauth_client.get(
        "/oauth/google_gmail/callback",
        params={"code": "fake-code", "state": state},
    )
    assert resp.status_code == 302
    location = resp.headers["location"]
    assert location.startswith("http://localhost:5173/dashboard/integrations")
    assert "provider=google_gmail" in location
    assert "status=connected" in location
    # token/secret 절대 노출 금지
    assert "fake-access-token" not in location
    assert "access_token" not in location
    assert "refresh_token" not in location


def test_callback_open_redirect_falls_back_to_json(oauth_client, monkeypatch):
    """허용되지 않은 origin이면 redirect하지 않고 JSON 응답으로 fallback."""
    from app.services.oauth.state import create_oauth_state

    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", "Cw3aE_Yp8jU2kZjKmCQ41eFPwJfH-tYxKXG2QJAcIIE=")
    monkeypatch.delenv("FRONTEND_ORIGIN", raising=False)
    monkeypatch.delenv("FRONTEND_ALLOWED_ORIGINS", raising=False)
    _stub_exchange_code(monkeypatch)

    state = create_oauth_state(
        "tenant-cb3",
        "slack",
        return_url="https://evil.example.com/steal",
    )

    resp = oauth_client.get(
        "/oauth/slack/callback",
        params={"code": "fake-code", "state": state},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "connected"
    assert data["provider"] == "slack"


def test_callback_jira_workspace_selection_required_redirect(oauth_client, monkeypatch):
    """Jira에서 workspace selection이 필요하면 redirect query에 reason 포함."""
    from app.services.oauth.state import create_oauth_state

    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", "Cw3aE_Yp8jU2kZjKmCQ41eFPwJfH-tYxKXG2QJAcIIE=")
    _stub_exchange_code(
        monkeypatch,
        scope="read:jira-user",
        raw={
            "accessible_resources": [
                {"id": "id-a", "name": "WorkspaceA", "url": "https://a.atlassian.net", "scopes": []},
                {"id": "id-b", "name": "WorkspaceB", "url": "https://b.atlassian.net", "scopes": []},
            ]
        },
    )

    state = create_oauth_state(
        "tenant-cb4",
        "jira",
        return_url="http://localhost:5173/dashboard/integrations",
    )

    resp = oauth_client.get(
        "/oauth/jira/callback",
        params={"code": "fake-code", "state": state},
    )
    assert resp.status_code == 302
    location = resp.headers["location"]
    assert "provider=jira" in location
    assert "workspace_selection_required=true" in location
    assert "reason=workspace_selection_required" in location


def test_status_jira_includes_workspace_selection_required(oauth_client):
    """Jira status는 workspace_selection_required와 workspace_id를 노출."""
    from app.models.tenant_integration import TenantIntegration, IntegrationStatus
    from app.repositories.tenant_integration_repo import upsert_integration

    upsert_integration(TenantIntegration(
        tenant_id="tenant-jira-status",
        provider="jira",
        status=IntegrationStatus.connected,
        external_workspace_id="cloud-1",
        external_workspace_name="Workspace 1",
        metadata={"workspace_selection_required": True, "accessible_resources": []},
    ))

    resp = oauth_client.get(
        "/oauth/jira/status",
        params={"tenant_id": "tenant-jira-status"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["workspace_selection_required"] is True
    assert data["workspace_id"] == "cloud-1"
