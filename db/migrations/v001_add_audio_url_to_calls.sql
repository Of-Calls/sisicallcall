-- ==============================================================================
-- v001_add_audio_url_to_calls.sql — calls 테이블 오디오 보관 컬럼 추가 (v3 M2)
-- 근거: db_schema.md §2.2 / screen_spec.md §12-D5 / api_spec.md v2 §3
-- ==============================================================================
-- 누적 환경(이미 calls 테이블이 존재하는 환경)용 ALTER 마이그레이션.
-- 신규 환경은 db/init/02_calls.sql 가 권위 소스이므로 본 파일 실행 불필요.
--
-- 컬럼 의미:
--   audio_url        — 객체 스토리지 mp3 URL (audio/{tenant_id}/{call_id}.mp3),
--                      미저장 통화는 NULL
--   audio_expires_at — 자동 삭제 예정 시각 (보관 7일 TTL, now() + 7 days)
--
-- 저장 스킵 조건 (둘 다 NULL 유지):
--   - calls.status = 'error' (파이프라인 오류 종료)
--   - tenant settings 의 personal_data=true / legal=true 정책 적용 시
--   - duration_sec < 5 (잘못 발신 등 의미 없는 통화)
-- ==============================================================================

ALTER TABLE calls
    ADD COLUMN IF NOT EXISTS audio_url        VARCHAR(255),
    ADD COLUMN IF NOT EXISTS audio_expires_at TIMESTAMPTZ;

-- audio_expires_at 도래 후 배치 정리(객체 삭제 + DB NULL 처리)에 사용
CREATE INDEX IF NOT EXISTS idx_calls_audio_expires_at
    ON calls(audio_expires_at)
    WHERE audio_expires_at IS NOT NULL;
