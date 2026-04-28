"""
TenantIntegration 모델 + TenantIntegrationRepository 테스트.

검증 범위:
  1.  TenantIntegration 기본 생성 (필수 필드, 기본값)
  2.  upsert_integration → get_integration 왕복
  3.  동일 tenant+provider 재등록 시 덮어쓰기 (upsert)
  4.  list_integrations — tenant 격리
  5.  mark_disconnected — status 변경
  6.  mark_disconnected — 없는 tenant 시 False 반환
  7.  update_tokens — access_token 갱신
  8.  update_tokens — 없는 tenant 시 False 반환
  9.  clear_integrations — 전체 삭제
  10. IntegrationStatus enum 값 검증
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta


@pytest.fixture(autouse=True)
def _clear_repo():
    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    yield
    tenant_integration_repo.clear_integrations()


# ── 1. TenantIntegration 기본 생성 ───────────────────────────────────────────

def test_tenant_integration_default_fields():
    from app.models.tenant_integration import TenantIntegration, IntegrationStatus

    ti = TenantIntegration(tenant_id="t1", provider="google_gmail")

    assert ti.tenant_id == "t1"
    assert ti.provider == "google_gmail"
    assert ti.status == IntegrationStatus.connected
    assert ti.scopes == []
    assert ti.token_type == "Bearer"
    assert ti.access_token_encrypted is None
    assert ti.refresh_token_encrypted is None
    assert ti.expires_at is None
    assert ti.id is not None
    assert ti.created_at is not None
    assert ti.updated_at is not None


# ── 2. upsert → get 왕복 ─────────────────────────────────────────────────────

def test_upsert_and_get_integration():
    from app.models.tenant_integration import TenantIntegration
    from app.repositories.tenant_integration_repo import upsert_integration, get_integration

    ti = TenantIntegration(
        tenant_id="t2",
        provider="slack",
        access_token_encrypted="enc-token-abc",
        external_account_email="admin@company.com",
    )
    upsert_integration(ti)

    result = get_integration("t2", "slack")
    assert result is not None
    assert result.tenant_id == "t2"
    assert result.provider == "slack"
    assert result.access_token_encrypted == "enc-token-abc"
    assert result.external_account_email == "admin@company.com"


# ── 3. 재등록 시 덮어쓰기 ────────────────────────────────────────────────────

def test_upsert_overwrites_existing():
    from app.models.tenant_integration import TenantIntegration
    from app.repositories.tenant_integration_repo import upsert_integration, get_integration

    ti1 = TenantIntegration(tenant_id="t3", provider="jira", access_token_encrypted="old-enc")
    ti2 = TenantIntegration(tenant_id="t3", provider="jira", access_token_encrypted="new-enc")

    upsert_integration(ti1)
    upsert_integration(ti2)

    result = get_integration("t3", "jira")
    assert result is not None
    assert result.access_token_encrypted == "new-enc"


# ── 4. list_integrations — tenant 격리 ───────────────────────────────────────

def test_list_integrations_tenant_isolation():
    from app.models.tenant_integration import TenantIntegration
    from app.repositories.tenant_integration_repo import upsert_integration, list_integrations

    upsert_integration(TenantIntegration(tenant_id="t4", provider="google_gmail"))
    upsert_integration(TenantIntegration(tenant_id="t4", provider="slack"))
    upsert_integration(TenantIntegration(tenant_id="OTHER", provider="google_gmail"))

    results = list_integrations("t4")
    assert len(results) == 2
    providers = {r.provider for r in results}
    assert providers == {"google_gmail", "slack"}


# ── 5. mark_disconnected ─────────────────────────────────────────────────────

def test_mark_disconnected_changes_status():
    from app.models.tenant_integration import TenantIntegration, IntegrationStatus
    from app.repositories.tenant_integration_repo import (
        upsert_integration, get_integration, mark_disconnected,
    )

    upsert_integration(TenantIntegration(tenant_id="t5", provider="slack"))
    ok = mark_disconnected("t5", "slack")

    assert ok is True
    result = get_integration("t5", "slack")
    assert result is not None
    assert result.status == IntegrationStatus.disconnected


# ── 6. mark_disconnected — 없는 tenant → False ───────────────────────────────

def test_mark_disconnected_nonexistent_returns_false():
    from app.repositories.tenant_integration_repo import mark_disconnected

    ok = mark_disconnected("no-such-tenant", "jira")
    assert ok is False


# ── 7. update_tokens ─────────────────────────────────────────────────────────

def test_update_tokens_refreshes_access_token():
    from app.models.tenant_integration import TenantIntegration, IntegrationStatus
    from app.repositories.tenant_integration_repo import (
        upsert_integration, get_integration, update_tokens,
    )

    upsert_integration(TenantIntegration(
        tenant_id="t6", provider="google_gmail",
        access_token_encrypted="old-enc",
    ))

    new_expires = datetime.utcnow() + timedelta(hours=1)
    ok = update_tokens(
        "t6", "google_gmail",
        access_token_encrypted="new-enc",
        refresh_token_encrypted="new-refresh-enc",
        expires_at=new_expires,
        status=IntegrationStatus.connected,
    )

    assert ok is True
    result = get_integration("t6", "google_gmail")
    assert result is not None
    assert result.access_token_encrypted == "new-enc"
    assert result.refresh_token_encrypted == "new-refresh-enc"
    assert result.status == IntegrationStatus.connected


# ── 8. update_tokens — 없는 tenant → False ───────────────────────────────────

def test_update_tokens_nonexistent_returns_false():
    from app.repositories.tenant_integration_repo import update_tokens

    ok = update_tokens(
        "no-tenant", "jira",
        access_token_encrypted="some-enc",
    )
    assert ok is False


# ── 9. clear_integrations ────────────────────────────────────────────────────

def test_clear_integrations_removes_all():
    from app.models.tenant_integration import TenantIntegration
    from app.repositories.tenant_integration_repo import (
        upsert_integration, list_integrations, clear_integrations,
    )

    upsert_integration(TenantIntegration(tenant_id="t7", provider="jira"))
    upsert_integration(TenantIntegration(tenant_id="t8", provider="slack"))

    clear_integrations()

    assert list_integrations("t7") == []
    assert list_integrations("t8") == []


# ── 10. IntegrationStatus enum 값 ────────────────────────────────────────────

def test_integration_status_enum_values():
    from app.models.tenant_integration import IntegrationStatus

    assert IntegrationStatus.connected == "connected"
    assert IntegrationStatus.disconnected == "disconnected"
    assert IntegrationStatus.expired == "expired"
    assert IntegrationStatus.error == "error"
