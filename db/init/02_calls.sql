-- ==============================================================================
-- 02_calls.sql — 통화 기록 및 구간별 레이턴시 / 브랜치 통계
-- db_schema.md §2.2
-- ==============================================================================

CREATE TABLE IF NOT EXISTS calls (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL REFERENCES tenants(id),
    twilio_call_sid VARCHAR(50) UNIQUE NOT NULL,       -- Twilio 고유 통화 ID
    caller_number   VARCHAR(20),                       -- 발신자 번호
    status          VARCHAR(20) DEFAULT 'in_progress'
                    CHECK (status IN (
                        'in_progress',  -- 통화 진행 중
                        'completed',    -- 정상 종료
                        'abandoned',    -- 고객이 먼저 끊음
                        'error'         -- 서버/파이프라인 오류로 비정상 종료
                    )),
    started_at      TIMESTAMPTZ DEFAULT now(),
    ended_at        TIMESTAMPTZ,
    duration_sec    INTEGER,                           -- 통화 시간 (초)
    audio_url       VARCHAR(255),                      -- v3 (M2): 객체 스토리지 mp3 URL, 미저장 시 NULL
    audio_expires_at TIMESTAMPTZ,                      -- v3 (M2): 자동 삭제 예정 시각 (보관 7일 TTL)
    latency_log     JSONB DEFAULT '{}',                -- 구간별 레이턴시 측정값
    branch_stats    JSONB DEFAULT '{}',                -- Cache/브랜치/Reviewer/Router 호출 카운터
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_calls_tenant_id ON calls(tenant_id);
CREATE INDEX IF NOT EXISTS idx_calls_started_at ON calls(started_at DESC);
-- v3 (M2): audio_expires_at 도래 후 배치 정리(객체 삭제 + DB NULL 처리) 대상 조회용
CREATE INDEX IF NOT EXISTS idx_calls_audio_expires_at
    ON calls(audio_expires_at)
    WHERE audio_expires_at IS NOT NULL;
