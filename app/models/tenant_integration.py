"""
TenantIntegration — SaaS 고객사별 OAuth 연동 정보 모델.

provider 예: "google_gmail", "google_calendar", "slack", "jira"
status 흐름: connected → expired → (refresh) → connected | error
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IntegrationStatus(str, Enum):
    connected = "connected"
    disconnected = "disconnected"
    expired = "expired"
    error = "error"


class TenantIntegration(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    provider: str

    status: IntegrationStatus = IntegrationStatus.connected
    scopes: list[str] = Field(default_factory=list)

    # Fernet-encrypted token bytes stored as str (base64 url-safe)
    access_token_encrypted: str | None = None
    refresh_token_encrypted: str | None = None
    token_type: str = "Bearer"
    expires_at: datetime | None = None

    # Provider-side account info
    external_account_id: str | None = None
    external_account_email: str | None = None
    external_workspace_id: str | None = None
    external_workspace_name: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
