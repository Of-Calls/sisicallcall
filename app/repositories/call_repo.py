"""calls 테이블 INSERT / UPDATE — 통화 시작 + 종료 시점 메타 기록.

통화 흐름 차단 방지 정책:
- 모든 함수 best-effort. asyncpg 예외는 흡수 + WARNING 로그만 남기고 None 반환.
- DB 다운 / connection 고갈 / schema 불일치 시에도 통화 응대 자체는 막지 않는다.

connection pool 미사용 — `_tenant_helpers.py` 와 동일하게 per-call asyncpg.connect.
풀 도입 시 본 모듈만 수정 (`_OPEN_ISSUES.md` 의 asyncpg pool 항목).
"""
import re

import asyncpg

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _is_uuid(value: str) -> bool:
    return bool(value and _UUID_RE.match(value))


async def insert_call(
    tenant_id: str,
    twilio_call_sid: str,
    caller_number: str | None = None,
) -> str | None:
    """calls 에 새 row INSERT 후 생성된 UUID 반환.

    실패 시 (DB 다운, 미등록 tenant, schema 위반 등) None 반환 — 호출자는 DB 추적 없이 진행.
    """
    if not _is_uuid(tenant_id):
        logger.warning("insert_call skip — invalid tenant_id=%s", tenant_id)
        return None
    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO calls (tenant_id, twilio_call_sid, caller_number, status)
                VALUES ($1::uuid, $2, $3, 'in_progress')
                RETURNING id
                """,
                tenant_id, twilio_call_sid, caller_number,
            )
            if row:
                db_call_id = str(row["id"])
                logger.info(
                    "calls INSERT db_call_id=%s twilio_call_sid=%s tenant_id=%s",
                    db_call_id, twilio_call_sid, tenant_id,
                )
                return db_call_id
        finally:
            await conn.close()
    except Exception as e:
        logger.warning(
            "insert_call failed twilio_call_sid=%s tenant_id=%s err=%s",
            twilio_call_sid, tenant_id, e,
        )
    return None


async def finalize_call(
    db_call_id: str,
    status: str,
    duration_sec: int | None = None,
) -> None:
    """통화 종료 시 status / ended_at / duration_sec 업데이트.

    status: 'completed' | 'abandoned' | 'error'  (스키마 CHECK 제약과 일치)
    """
    if not _is_uuid(db_call_id):
        return
    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            await conn.execute(
                """
                UPDATE calls
                SET status = $1, ended_at = now(), duration_sec = $2
                WHERE id = $3::uuid
                """,
                status, duration_sec, db_call_id,
            )
            logger.info(
                "calls UPDATE db_call_id=%s status=%s duration_sec=%s",
                db_call_id, status, duration_sec,
            )
        finally:
            await conn.close()
    except Exception as e:
        logger.warning(
            "finalize_call failed db_call_id=%s status=%s err=%s",
            db_call_id, status, e,
        )
