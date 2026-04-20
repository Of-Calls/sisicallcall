-- ==============================================================================
-- 07_knn_intents.sql — KNN Router 학습용 intent 예시 문장
-- db_schema.md §2.7
-- ==============================================================================
-- 관리자가 예시 문장 추가/수정 시 BGE-M3 임베딩을 즉시 계산하여 함께 저장.
-- 서버 시작 시 또는 updated_at 변경 감지 시 메모리 내 KNN 재구성.
-- tenant_id별 독립 인덱스 (멀티테넌시).
-- ==============================================================================

CREATE TABLE IF NOT EXISTS knn_intents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL REFERENCES tenants(id),
    intent_label    VARCHAR(100) NOT NULL,             -- 'intent_faq_hours', 'intent_task_reservation' 등
    example_text    TEXT NOT NULL,                     -- 예시 발화 문장
    embedding       REAL[] NOT NULL,                   -- BGE-M3 임베딩 (문장 추가/수정 시 즉시 계산)
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()          -- KNN 재구성 트리거 기준
);

CREATE INDEX IF NOT EXISTS idx_knn_intents_tenant_id ON knn_intents(tenant_id);
CREATE INDEX IF NOT EXISTS idx_knn_intents_updated_at ON knn_intents(updated_at DESC);
