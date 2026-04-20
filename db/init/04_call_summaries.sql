-- ==============================================================================
-- 04_call_summaries.sql — 통화 요약 (동기/비동기 하이브리드)
-- db_schema.md §2.4
-- ==============================================================================
-- call_id UNIQUE — 통화당 1개 레코드. 동기 모드 INSERT 후 비동기 모드 UPDATE.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS call_summaries (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id           UUID UNIQUE NOT NULL REFERENCES calls(id),
    tenant_id         UUID NOT NULL REFERENCES tenants(id),
    summary_short     TEXT NOT NULL,                     -- 200자 이내, 동기 모드의 핵심 출력
    summary_detailed  TEXT,                              -- 500~1000자, 비동기 모드에서만 채움
    customer_intent   VARCHAR(200),                      -- 핵심 문의 내용
    customer_emotion  VARCHAR(20)
                      CHECK (customer_emotion IN (
                          'positive', 'neutral', 'negative', 'angry'
                      )),
    resolution_status VARCHAR(20)
                      CHECK (resolution_status IN (
                          'resolved',    -- 통화 내 해결 완료
                          'escalated',   -- 상담원 연결/콜백 예약
                          'abandoned'    -- 고객이 중간 이탈
                      )),
    keywords          JSONB DEFAULT '[]',                -- 핵심 키워드 배열 (3~10개)
    handoff_notes     TEXT,                              -- Escalation 시 상담원용 인수인계 메모
    generation_mode   VARCHAR(10) NOT NULL
                      CHECK (generation_mode IN ('sync', 'async')),
    model_used        VARCHAR(50) NOT NULL,              -- 'gpt-4o-mini' | 'gpt-4o' 등
    created_at        TIMESTAMPTZ DEFAULT now(),
    updated_at        TIMESTAMPTZ DEFAULT now()          -- 동기 → 비동기 UPDATE 시 갱신
);

CREATE INDEX IF NOT EXISTS idx_call_summaries_tenant_id ON call_summaries(tenant_id);
CREATE INDEX IF NOT EXISTS idx_call_summaries_customer_emotion ON call_summaries(customer_emotion);
CREATE INDEX IF NOT EXISTS idx_call_summaries_resolution_status ON call_summaries(resolution_status);
CREATE INDEX IF NOT EXISTS idx_call_summaries_created_at ON call_summaries(created_at DESC);
