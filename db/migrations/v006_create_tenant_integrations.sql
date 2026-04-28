-- ==============================================================================
-- v006_create_tenant_integrations.sql — tenant 별 외부 연동 OAuth 토큰 테이블 신설
-- 근거: post_call 팀 요청 (MCP 토큰 영속화 — Gmail / Calendar / Slack / Jira)
-- ==============================================================================
-- 누적 환경(이미 PostgreSQL 이 떠 있는 환경)용 마이그레이션.
-- 신규 환경은 db/init/11_tenant_integrations.sql 이 권위 소스 — 본 파일 실행 불필요.
--
-- 보안:
--   access_token_encrypted / refresh_token_encrypted 는 애플리케이션 레이어에서
--   암복호화 (env / KMS / Vault — 별도 결정). DB 자체는 평문 컬럼.
--
-- UNIQUE 제약:
--   (tenant_id, provider) — tenant 당 provider 1개. 멀티 계정 지원이 필요해지면
--   external_account_id 까지 확장한 UNIQUE 로 ALTER (후처리팀 코드 합류 후 결정).
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
