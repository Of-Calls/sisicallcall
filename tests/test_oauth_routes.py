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
    from app.services.oauth.state import clear_oauth_states
    from app.repositories.tenant_integration_repo import tenant_integration_repo
    clear_oauth_states()
    tenant_integration_repo.clear_integrations()
    yield
    clear_oauth_states()
    tenant_integration_repo.clear_integrations()


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
