"""
TenantIntegrationRepository — in-memory 구현.

키: (tenant_id, provider)
실제 배포 시 PostgreSQL 구현체로 교체 예정.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from app.models.tenant_integration import IntegrationStatus, TenantIntegration
from app.utils.logger import get_logger

logger = get_logger(__name__)

_store: dict[tuple[str, str], TenantIntegration] = {}


class TenantIntegrationRepository:
    def upsert_integration(self, integration: TenantIntegration) -> TenantIntegration:
        key = (integration.tenant_id, integration.provider)
        integration.updated_at = datetime.utcnow()
        _store[key] = integration
        logger.debug(
            "TenantIntegration upserted tenant_id=%s provider=%s status=%s",
            integration.tenant_id, integration.provider, integration.status,
        )
        return integration

    def get_integration(self, tenant_id: str, provider: str) -> TenantIntegration | None:
        return _store.get((tenant_id, provider))

    def list_integrations(self, tenant_id: str) -> list[TenantIntegration]:
        return [v for (t, _), v in _store.items() if t == tenant_id]

    def mark_disconnected(self, tenant_id: str, provider: str) -> bool:
        integration = _store.get((tenant_id, provider))
        if integration is None:
            return False
        integration.status = IntegrationStatus.disconnected
        integration.updated_at = datetime.utcnow()
        return True

    def update_tokens(
        self,
        tenant_id: str,
        provider: str,
        *,
        access_token_encrypted: str,
        refresh_token_encrypted: str | None = None,
        expires_at: datetime | None = None,
        status: IntegrationStatus = IntegrationStatus.connected,
    ) -> bool:
        integration = _store.get((tenant_id, provider))
        if integration is None:
            return False
        integration.access_token_encrypted = access_token_encrypted
        if refresh_token_encrypted is not None:
            integration.refresh_token_encrypted = refresh_token_encrypted
        if expires_at is not None:
            integration.expires_at = expires_at
        integration.status = status
        integration.updated_at = datetime.utcnow()
        return True

    def clear_integrations(self) -> None:
        _store.clear()


# 모듈 레벨 싱글턴
tenant_integration_repo = TenantIntegrationRepository()


# ── 편의 함수 ─────────────────────────────────────────────────────────────────

def upsert_integration(integration: TenantIntegration) -> TenantIntegration:
    return tenant_integration_repo.upsert_integration(integration)


def get_integration(tenant_id: str, provider: str) -> TenantIntegration | None:
    return tenant_integration_repo.get_integration(tenant_id, provider)


def list_integrations(tenant_id: str) -> list[TenantIntegration]:
    return tenant_integration_repo.list_integrations(tenant_id)


def mark_disconnected(tenant_id: str, provider: str) -> bool:
    return tenant_integration_repo.mark_disconnected(tenant_id, provider)


def update_tokens(
    tenant_id: str,
    provider: str,
    *,
    access_token_encrypted: str,
    refresh_token_encrypted: str | None = None,
    expires_at: datetime | None = None,
    status: IntegrationStatus = IntegrationStatus.connected,
) -> bool:
    return tenant_integration_repo.update_tokens(
        tenant_id, provider,
        access_token_encrypted=access_token_encrypted,
        refresh_token_encrypted=refresh_token_encrypted,
        expires_at=expires_at,
        status=status,
    )


def clear_integrations() -> None:
    tenant_integration_repo.clear_integrations()
