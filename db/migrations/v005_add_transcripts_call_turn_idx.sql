-- v005_add_transcripts_call_turn_idx.sql
-- transcripts(call_id, turn_index) 복합 인덱스 추가.
--
-- 배경:
--   post_call 이 향후 transcripts 를 SELECT ... WHERE call_id=$1 ORDER BY turn_index
--   형태로 조회 예정. 기존 idx_transcripts_call_id 는 prefix 만 커버해 ORDER BY 시
--   추가 sort 비용 발생.
--
-- 비고:
--   신규 복합 인덱스가 (call_id) prefix 도 커버하므로 idx_transcripts_call_id 는
--   잉여이지만, 기존 운영 환경 영향 최소화를 위해 보수적으로 유지(drop 안 함).
--   db/init/03_transcripts.sql 에도 동일 인덱스가 추가되어 신규 환경 일관성 확보.

CREATE INDEX IF NOT EXISTS idx_transcripts_call_id_turn
    ON transcripts(call_id, turn_index);
