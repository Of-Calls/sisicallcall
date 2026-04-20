-- ==============================================================================
-- 08_rag_documents.sql — RAG 업로드 문서 메타데이터
-- db_schema.md §2.8
-- ==============================================================================
-- 벡터 본체는 ChromaDB에 저장. 이 테이블은 관리 목적의 메타데이터만 보관.
-- 소프트 삭제 (deleted_at) — ChromaDB 동시 삭제 필수.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS rag_documents (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id         UUID NOT NULL REFERENCES tenants(id),
    file_name         VARCHAR(255) NOT NULL,           -- 업로드 파일명
    file_type         VARCHAR(10) NOT NULL
                      CHECK (file_type IN ('pdf', 'faq')),
    chunk_count       INTEGER,                         -- 생성된 청크 수
    status            VARCHAR(20) DEFAULT 'processing'
                      CHECK (status IN ('processing', 'ready', 'failed')),
    chroma_collection VARCHAR(100),                    -- 연결된 ChromaDB 컬렉션명
    uploaded_at       TIMESTAMPTZ DEFAULT now(),
    indexed_at        TIMESTAMPTZ,                     -- ChromaDB 인덱싱 완료 시각
    deleted_at        TIMESTAMPTZ DEFAULT NULL         -- 소프트 삭제 — NULL이면 정상
);

CREATE INDEX IF NOT EXISTS idx_rag_documents_tenant_id ON rag_documents(tenant_id);
-- partial index: 정상 문서만 빠르게 조회
CREATE INDEX IF NOT EXISTS idx_rag_documents_deleted_at ON rag_documents(deleted_at)
    WHERE deleted_at IS NULL;
