-- ==============================================================================
-- 06_face_embeddings.sql — ArcFace 512d 임베딩 (M3+ 금융 버티컬)
-- db_schema.md §2.6
-- ==============================================================================
-- 얼굴 인증은 Tier 3+ 기능. M1 단계에서는 스키마만 생성, 빈 테이블로 유지.
-- 고객당 1개 임베딩만 유지 (재등록 시 UPDATE로 덮어쓰기).
-- ==============================================================================

CREATE TABLE IF NOT EXISTS face_embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL REFERENCES tenants(id),
    customer_ref    VARCHAR(100) NOT NULL,             -- 고객 식별자 (고객사 내부 ID)
    embedding       REAL[] NOT NULL,                   -- ArcFace 512d 임베딩 벡터
    registered_at   TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(tenant_id, customer_ref)
);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_tenant_id ON face_embeddings(tenant_id);
