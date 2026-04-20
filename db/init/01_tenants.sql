-- ==============================================================================
-- 01_tenants.sql — 고객사 정보 및 Twilio 번호 매핑
-- db_schema.md §2.1
-- ==============================================================================
-- Twilio Webhook 수신 시 전화번호 → tenant_id 분기의 기준이 됩니다.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS tenants (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(100) NOT NULL,             -- 고객사명
    twilio_number   VARCHAR(20) UNIQUE NOT NULL,       -- Twilio 수신 번호 (+821012345678)
    industry        VARCHAR(50)
                    CHECK (industry IN ('hospital', 'restaurant', 'finance', 'appliance')),
    plan            VARCHAR(20) DEFAULT 'basic'
                    CHECK (plan IN ('basic', 'vertical')),
    is_active       BOOLEAN DEFAULT TRUE,
    settings        JSONB DEFAULT '{}',                -- 업종별 커스텀 설정 (임계값 등)
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tenants_twilio_number ON tenants(twilio_number);
