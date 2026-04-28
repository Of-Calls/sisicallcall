-- ==============================================================================
-- 11_tenant_integrations.sql — tenant 별 외부 연동 OAuth 토큰 저장
-- db_schema.md §2.11 (예정)
-- ==============================================================================
-- post_call Agent 의 MCP 가 Gmail / Calendar / Slack / Jira 등 외부 시스템에
-- OAuth 인증 후 access/refresh token 으로 접근하기 위한 보관 테이블.
-- 토큰 자체의 암복호화는 애플리케이션 레이어 책임 (스키마는 컬럼명으로만 의도 표시).
-- ==============================================================================

CREATE TABLE IF NOT EXISTS tenant_integrations (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                UUID NOT NULL REFERENCES tenants(id),

    provider                 VARCHAR(50) NOT NULL,
    -- google, gmail, calendar, slack, jira (CHECK 미적용 — 새 provider 자유 추가)

    status                   VARCHAR(20) NOT NULL DEFAULT 'connected'
                             CHECK (status IN ('connected', 'disconnected', 'expired', 'error')),

    scopes                   JSONB DEFAULT '[]',

    access_token_encrypted   TEXT,
    refresh_token_encrypted  TEXT,
    token_type               VARCHAR(50),                  -- 보통 'Bearer'
    expires_at               TIMESTAMPTZ,                  -- access_token 만료 — refresh_token 으로 갱신

    external_account_id      VARCHAR(255),
    external_account_email   VARCHAR(255),
    external_workspace_id    VARCHAR(255),
    external_workspace_name  VARCHAR(255),

    metadata                 JSONB DEFAULT '{}',

    created_at               TIMESTAMPTZ DEFAULT now(),
    updated_at               TIMESTAMPTZ DEFAULT now(),

    UNIQUE (tenant_id, provider)
);

CREATE INDEX IF NOT EXISTS idx_tenant_integrations_tenant_id ON tenant_integrations(tenant_id);
