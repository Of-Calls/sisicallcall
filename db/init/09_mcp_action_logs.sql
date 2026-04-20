-- ==============================================================================
-- 09_mcp_action_logs.sql — MCP 연동 처리 이력
-- db_schema.md §2.9
-- ==============================================================================
-- Gmail / Calendar / 임시 회사 DB 액션 모두 기록.
-- Task 브랜치 tool calling 결과가 1:1 기록되어 실패 분석 기준으로 사용.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS mcp_action_logs (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id          UUID NOT NULL REFERENCES calls(id),
    tenant_id        UUID NOT NULL REFERENCES tenants(id),
    action_type      VARCHAR(30) NOT NULL
                     CHECK (action_type IN ('gmail', 'calendar', 'company_db')),
    action_detail    VARCHAR(100),                     -- '예약_insert', '요약_메일_발송' 등
    status           VARCHAR(10) NOT NULL
                     CHECK (status IN ('success', 'fail')),
    request_payload  JSONB DEFAULT '{}',               -- MCP 요청 내용
    response_payload JSONB DEFAULT '{}',               -- MCP 응답 내용
    error_message    TEXT,                             -- 실패 시 에러 메시지
    executed_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mcp_action_logs_call_id ON mcp_action_logs(call_id);
CREATE INDEX IF NOT EXISTS idx_mcp_action_logs_tenant_id ON mcp_action_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mcp_action_logs_action_type ON mcp_action_logs(action_type);
