-- ==============================================================================
-- v002_add_cluster_xy_to_voc_analyses.sql — voc_analyses UMAP 2D 좌표 (v3 M2)
-- 근거: db_schema.md §2.5 / screen_spec.md §12-D11 / api_spec.md v2 §5
-- ==============================================================================
-- K-means 배치 스크립트(app/agents/voc/cluster.py)가 클러스터 재계산 시
-- UMAP(n_components=2) 2D 투영을 함께 저장.
-- 대시보드 산점도(/analytics/clusters)의 점 좌표로 소비.
--
-- 값 예시: {"x": 0.124, "y": -0.873}
-- 미실행 / 재계산 전 레코드는 NULL.
--
-- ⚠️ 무결성 규약: cluster_label 과 cluster_xy 는 반드시 같은 배치 실행에서
-- 동시 UPDATE — 불일치 상태 금지 (한쪽만 NULL 허용 안 함).
-- ==============================================================================

ALTER TABLE voc_analyses
    ADD COLUMN IF NOT EXISTS cluster_xy JSONB;
