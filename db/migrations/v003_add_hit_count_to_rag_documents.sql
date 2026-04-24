-- ==============================================================================
-- v003_add_hit_count_to_rag_documents.sql — RAG 문서 힛 카운트 (v3 M2)
-- 근거: db_schema.md §2.8 / screen_spec.md §12-D9 / api_spec.md v2 §6
-- ==============================================================================
-- 누적 환경(이미 rag_documents 테이블이 존재하는 환경)용 ALTER 마이그레이션.
-- 신규 환경은 db/init/08_rag_documents.sql 가 권위 소스이므로 본 파일 실행 불필요.
--
-- 컬럼 의미:
--   hit_count   — RAG 검색 응답에 포함된 누적 횟수 (기본 0)
--   last_hit_at — 가장 최근에 참조된 시각
--
-- 갱신 주체: rag_node 가 검색 결과로 응답 생성 시 UPDATE …
--           SET hit_count = hit_count + 1, last_hit_at = now() WHERE id = …
-- 인덱스: 대시보드 RAG 문서 정렬(자주 참조되는 문서 우선)에 사용.
-- ==============================================================================

ALTER TABLE rag_documents
    ADD COLUMN IF NOT EXISTS hit_count   INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_hit_at TIMESTAMPTZ;

-- soft delete 미적용 문서만 대상으로 정렬 — partial index
CREATE INDEX IF NOT EXISTS idx_rag_documents_hit_count
    ON rag_documents(hit_count DESC)
    WHERE deleted_at IS NULL;
