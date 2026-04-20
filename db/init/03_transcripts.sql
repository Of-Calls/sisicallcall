-- ==============================================================================
-- 03_transcripts.sql — 발화 단위 transcript
-- db_schema.md §2.3
-- ==============================================================================
-- tenant_id 컬럼 없음 — calls JOIN으로 tenant 격리
-- ==============================================================================

CREATE TABLE IF NOT EXISTS transcripts (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id           UUID NOT NULL REFERENCES calls(id),
    turn_index        INTEGER NOT NULL,                  -- 발화 순서 (0부터 시작)
    speaker           VARCHAR(10) NOT NULL
                      CHECK (speaker IN ('customer', 'agent')),
    text              TEXT NOT NULL,                     -- 발화 텍스트
    response_path     VARCHAR(20)
                      CHECK (response_path IN (
                          'cache',       -- Gate 1 Semantic Cache hit
                          'faq',         -- FAQ 브랜치
                          'task',        -- Task 브랜치
                          'auth',        -- Auth 브랜치
                          'escalation'   -- Escalation 브랜치
                      )),                                -- agent 발화만 해당, customer 발화는 NULL
    reviewer_applied  BOOLEAN DEFAULT FALSE,             -- 조건부 Reviewer 호출 여부
    reviewer_verdict  VARCHAR(10)
                      CHECK (reviewer_verdict IN ('pass', 'revise')), -- Reviewer 호출 시만 값 존재
    is_barge_in       BOOLEAN DEFAULT FALSE,             -- barge-in으로 발생한 발화 여부
    spoken_at         TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_transcripts_call_id ON transcripts(call_id);
