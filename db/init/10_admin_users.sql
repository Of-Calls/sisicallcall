-- ==============================================================================
-- 10_admin_users.sql — 관리자 대시보드 로그인 계정 (v3 신규, M1)
-- db_schema.md §2.10
-- ==============================================================================
-- screen_spec.md §12-D1 / api_spec.md §13 확정에 따라 M1 에 신설.
-- 고객 얼굴 인증(face_embeddings, §2.6) 과는 완전히 분리된 도메인.
-- 본 테이블은 SaaS 관리자(기업 담당자) 로그인 전용.
--
-- 정책 요약:
--   - role 은 M1 에서 'owner' 단일만 사용 (한 tenant 당 오너 1개)
--   - 'manager' / 'agent' 는 M2 권한 분리 시 활성화 (별도 members 테이블 만들지 않음)
--   - password_hash 는 반드시 bcrypt cost 12 (app/services/auth/password.py)
--   - 로그인 흐름: POST /admin/auth/login → Redis admin_session:{session_id} 저장
--     + last_login_at UPDATE (db_schema.md §3.10)
--   - DELETE 정책: soft delete 미적용. tenant 탈퇴 시 함께 DELETE
--     (FK CASCADE 미설정 — 애플리케이션 트랜잭션 처리)
-- ==============================================================================

CREATE TABLE IF NOT EXISTS admin_users (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id      UUID NOT NULL REFERENCES tenants(id),
    email          VARCHAR(255) UNIQUE NOT NULL,        -- 전역 UNIQUE (로그인 식별자)
    password_hash  VARCHAR(255) NOT NULL,               -- bcrypt cost 12
    role           VARCHAR(20) DEFAULT 'owner'
                   CHECK (role IN ('owner', 'manager', 'agent')),
    last_login_at  TIMESTAMPTZ,
    created_at     TIMESTAMPTZ DEFAULT now(),
    updated_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_admin_users_tenant_id ON admin_users(tenant_id);
-- 대소문자 구분 없는 이메일 UNIQUE — 'User@x.com' / 'user@x.com' 중복 차단
CREATE UNIQUE INDEX IF NOT EXISTS idx_admin_users_email_lower ON admin_users(LOWER(email));
