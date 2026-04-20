-- ==============================================================================
-- 05_voc_analyses.sql — VOC 분석 통합 결과 (M2)
-- db_schema.md §2.5
-- ==============================================================================
-- 3개 서브 에이전트(감정/의도/우선순위) 병렬 실행 결과 upsert.
-- M2에서 최초 활성화 — M1 단계에서는 빈 테이블로 유지.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS voc_analyses (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id           UUID UNIQUE NOT NULL REFERENCES calls(id),
    tenant_id         UUID NOT NULL REFERENCES tenants(id),
    sentiment_result  JSONB DEFAULT '{}',                -- 감정 서브 에이전트 출력
    intent_result     JSONB DEFAULT '{}',                -- 의도 서브 에이전트 출력
    priority_result   JSONB DEFAULT '{}',                -- 우선순위 서브 에이전트 출력
    partial_success   BOOLEAN DEFAULT FALSE,             -- 일부 서브 실패 여부
    failed_subagents  JSONB DEFAULT '[]',                -- 실패한 서브 이름 배열
    cluster_label     INTEGER,                           -- K-means 배치 작업 결과 (intent_result 기반)
    created_at        TIMESTAMPTZ DEFAULT now(),
    updated_at        TIMESTAMPTZ DEFAULT now()          -- 재시도로 partial_success 해소 시 갱신
);

CREATE INDEX IF NOT EXISTS idx_voc_analyses_tenant_id ON voc_analyses(tenant_id);
CREATE INDEX IF NOT EXISTS idx_voc_analyses_call_id ON voc_analyses(call_id);
CREATE INDEX IF NOT EXISTS idx_voc_analyses_cluster_label ON voc_analyses(cluster_label);
CREATE INDEX IF NOT EXISTS idx_voc_analyses_created_at ON voc_analyses(created_at DESC);
