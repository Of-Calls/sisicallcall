-- ==============================================================================
-- 09_mcp_action_logs.sql - MCP action execution log
-- ==============================================================================
-- Post-call action idempotency and audit log.
-- call_id is intentionally TEXT so both UUID call ids and Twilio SIDs can be used.
-- tenant_id is TEXT because MCP/OAuth integrations currently use tenant labels too.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS mcp_action_logs (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id          TEXT NOT NULL,
    tenant_id        TEXT,
    action_type      TEXT NOT NULL,
    tool_name        TEXT NOT NULL,
    request_payload  JSONB DEFAULT '{}'::jsonb,
    response_payload JSONB DEFAULT '{}'::jsonb,
    status           TEXT NOT NULL,
    external_id      TEXT,
    error_message    TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
DECLARE
    constraint_record RECORD;
BEGIN
    FOR constraint_record IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'mcp_action_logs'::regclass
          AND contype IN ('f', 'c')
    LOOP
        EXECUTE format(
            'ALTER TABLE mcp_action_logs DROP CONSTRAINT IF EXISTS %I',
            constraint_record.conname
        );
    END LOOP;
END $$;

ALTER TABLE mcp_action_logs
    ADD COLUMN IF NOT EXISTS tool_name TEXT,
    ADD COLUMN IF NOT EXISTS external_id TEXT,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

ALTER TABLE mcp_action_logs
    ALTER COLUMN call_id TYPE TEXT USING call_id::text,
    ALTER COLUMN tenant_id DROP NOT NULL,
    ALTER COLUMN tenant_id TYPE TEXT USING tenant_id::text,
    ALTER COLUMN action_type TYPE TEXT USING action_type::text,
    ALTER COLUMN status TYPE TEXT USING status::text,
    ALTER COLUMN request_payload SET DEFAULT '{}'::jsonb,
    ALTER COLUMN response_payload SET DEFAULT '{}'::jsonb;

UPDATE mcp_action_logs
SET tool_name = COALESCE(tool_name, action_type, 'unknown')
WHERE tool_name IS NULL;

ALTER TABLE mcp_action_logs
    ALTER COLUMN tool_name SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_mcp_action_logs_call_id
ON mcp_action_logs(call_id);

CREATE INDEX IF NOT EXISTS idx_mcp_action_logs_tenant_id
ON mcp_action_logs(tenant_id);

CREATE INDEX IF NOT EXISTS idx_mcp_action_logs_idempotency
ON mcp_action_logs(call_id, action_type, tool_name, status);
