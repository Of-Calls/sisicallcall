"""
TenantIntegrationRepository — in-memory + 선택적 file persistence 구현.

저장 백엔드는 TENANT_INTEGRATION_STORAGE 환경변수로 선택한다:
  memory  (기본): in-memory dict. 서버 재시작 시 데이터 소멸.
  file:           JSON 파일 persistence. 서버 재시작 후에도 유지.
                  TENANT_INTEGRATION_FILE_PATH로 경로 지정
                  (기본: .local/tenant_integrations.json)

── 파일 저장 형식 ─────────────────────────────────────────────────────────────
{
  "{tenant_id}::{provider}": { ...TenantIntegration 필드... }
}
access_token_encrypted / refresh_token_encrypted 는 이미 Fernet 암호화된
문자열만 저장한다. 평문 토큰은 절대 파일에 기록하지 않는다.

── PostgreSQL 전환 TODO ────────────────────────────────────────────────────────
운영 전환 시 이 파일을 asyncpg 기반 구현체로 교체한다.
인터페이스(메서드 시그니처)는 변경하지 않는다:
  upsert_integration(integration) → TenantIntegration
  get_integration(tenant_id, provider) → TenantIntegration | None
  list_integrations(tenant_id) → list[TenantIntegration]
  mark_disconnected(tenant_id, provider) → bool
  update_tokens(tenant_id, provider, *, ...) → bool
  clear_integrations() → None

PostgreSQL 스키마 예시:
  CREATE TABLE tenant_integrations (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               TEXT NOT NULL,
    provider                TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'connected',
    scopes                  TEXT[] DEFAULT '{}',
    access_token_encrypted  TEXT,
    refresh_token_encrypted TEXT,
    token_type              TEXT DEFAULT 'Bearer',
    expires_at              TIMESTAMPTZ,
    external_account_id     TEXT,
    external_account_email  TEXT,
    external_workspace_id   TEXT,
    external_workspace_name TEXT,
    metadata                JSONB DEFAULT '{}',
    created_at              TIMESTAMPTZ DEFAULT now(),
    updated_at              TIMESTAMPTZ DEFAULT now(),
    UNIQUE (tenant_id, provider)
  );
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models.tenant_integration import IntegrationStatus, TenantIntegration
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DATETIME_FMT = "%Y-%m-%dT%H:%M:%S.%f"

# ── 직렬화 헬퍼 ───────────────────────────────────────────────────────────────

def _to_dict(ti: TenantIntegration) -> dict[str, Any]:
    d = ti.model_dump()
    for k in ("expires_at", "created_at", "updated_at"):
        if d[k] is not None:
            d[k] = d[k].strftime(_DATETIME_FMT)
    d["status"] = d["status"].value if hasattr(d["status"], "value") else d["status"]
    return d


def _from_dict(d: dict[str, Any]) -> TenantIntegration:
    for k in ("expires_at", "created_at", "updated_at"):
        if d.get(k):
            d[k] = datetime.strptime(d[k], _DATETIME_FMT)
    return TenantIntegration(**d)


# ── 파일 I/O ──────────────────────────────────────────────────────────────────

def _file_path() -> Path:
    raw = os.getenv("TENANT_INTEGRATION_FILE_PATH", ".local/tenant_integrations.json")
    return Path(raw)


def _load_file(path: Path) -> dict[str, TenantIntegration]:
    if not path.exists():
        return {}
    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return {k: _from_dict(v) for k, v in raw.items()}
    except Exception as exc:
        logger.error("tenant_integration 파일 로드 실패 path=%s err=%s", path, exc)
        return {}


def _save_file(path: Path, store: dict[str, TenantIntegration]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = {k: _to_dict(v) for k, v in store.items()}
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.error("tenant_integration 파일 저장 실패 path=%s err=%s", path, exc)


# ── Repository ────────────────────────────────────────────────────────────────

class TenantIntegrationRepository:
    """
    기본 백엔드: in-memory.
    TENANT_INTEGRATION_STORAGE=file 설정 시 JSON 파일로 persist.
    """

    def __init__(self, *, storage: str | None = None) -> None:
        self._storage = storage or os.getenv("TENANT_INTEGRATION_STORAGE", "memory")
        # in-memory store (file 모드에서도 캐시로 사용)
        self._store: dict[str, TenantIntegration] = {}
        if self._storage == "file":
            self._store = _load_file(_file_path())
            logger.info(
                "TenantIntegrationRepo file mode path=%s loaded=%d",
                _file_path(), len(self._store),
            )

    # ── 내부 ─────────────────────────────────────────────────────────────────

    def _key(self, tenant_id: str, provider: str) -> str:
        return f"{tenant_id}::{provider}"

    def _persist(self) -> None:
        if self._storage == "file":
            _save_file(_file_path(), self._store)

    # ── 공개 인터페이스 ────────────────────────────────────────────────────────

    def upsert_integration(self, integration: TenantIntegration) -> TenantIntegration:
        key = self._key(integration.tenant_id, integration.provider)
        integration.updated_at = datetime.utcnow()
        self._store[key] = integration
        self._persist()
        logger.debug(
            "TenantIntegration upserted tenant_id=%s provider=%s status=%s",
            integration.tenant_id, integration.provider, integration.status,
        )
        return integration

    def get_integration(self, tenant_id: str, provider: str) -> TenantIntegration | None:
        return self._store.get(self._key(tenant_id, provider))

    def list_integrations(self, tenant_id: str) -> list[TenantIntegration]:
        prefix = f"{tenant_id}::"
        return [v for k, v in self._store.items() if k.startswith(prefix)]

    def mark_disconnected(self, tenant_id: str, provider: str) -> bool:
        key = self._key(tenant_id, provider)
        integration = self._store.get(key)
        if integration is None:
            return False
        integration.status = IntegrationStatus.disconnected
        integration.updated_at = datetime.utcnow()
        self._persist()
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
        key = self._key(tenant_id, provider)
        integration = self._store.get(key)
        if integration is None:
            return False
        integration.access_token_encrypted = access_token_encrypted
        if refresh_token_encrypted is not None:
            integration.refresh_token_encrypted = refresh_token_encrypted
        if expires_at is not None:
            integration.expires_at = expires_at
        integration.status = status
        integration.updated_at = datetime.utcnow()
        self._persist()
        return True

    def clear_integrations(self) -> None:
        self._store.clear()
        if self._storage == "file":
            path = _file_path()
            if path.exists():
                path.write_text("{}", encoding="utf-8")


# ── 모듈 레벨 싱글턴 ──────────────────────────────────────────────────────────
# TENANT_INTEGRATION_STORAGE 환경변수를 읽어 백엔드 결정.
# 테스트에서는 각 테스트가 직접 TenantIntegrationRepository(storage="memory") 인스턴스를
# 생성하거나, clear_integrations()로 격리하면 된다.

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
