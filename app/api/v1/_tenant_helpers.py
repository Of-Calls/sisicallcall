"""테넌트 식별/조회 + 인사말 헬퍼 — call.py 에서 분리 (2026-04-24).

WebSocket 핸들러 본문이 비대해져 단일 책임 원칙 위배 → Tenant DB 조회 + greeting 처리만
이 모듈로 분리. call.py 는 라우팅 + 오디오 파이프라인 흐름에 집중한다.

외부에서 사용 (call.py):
    from app.api.v1._tenant_helpers import (
        resolve_tenant_id, get_tenant_name, get_greeting,
    )

DB 접근은 모두 asyncpg per-call connection 사용. connection pool 도입 시 본 모듈만 수정하면 됨
(_OPEN_ISSUES.md 참조).
"""
import json as _json
import re

import asyncpg

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 인사말 기본값 — tenants.settings JSONB 에 greeting/offhours_greeting 없을 때 사용
DEFAULT_GREETING = "안녕하세요, 고객센터입니다. 무엇을 도와드릴까요?"
DEFAULT_OFFHOURS_GREETING = (
    "안녕하세요, 고객센터입니다. "
    "현재 상담원 운영 시간이 아니지만 기본적인 문의는 도와드릴 수 있습니다. "
    "무엇을 도와드릴까요?"
)

# UUID 검증 — _resolve_tenant_id 실패 시 SIP URI 등 raw 값이 넘어오는 경우 DB 조회 차단
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# 테넌트 이름 in-process 캐시 — 통화 시작 시 1회 조회 후 재사용.
# (intent_router_llm 시스템 프롬프트에 동적 주입되므로 통화마다 필요)
_tenant_name_cache: dict[str, str] = {}


async def resolve_tenant_id(to_number: str) -> str:
    """Twilio To 필드(전화번호 또는 SIP URI)로 tenant UUID 조회.

    1차: raw 값 그대로 매칭
    2차: SIP URI(`sip:5@host`) → user part(`5`) 매칭
    미등록 시 raw 값을 그대로 반환하고 경고 로그 출력.
    """
    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                "SELECT id, name FROM tenants WHERE twilio_number = $1", to_number
            )
            if row:
                logger.info(
                    "tenant 매칭 (raw) to=%s → id=%s name=%s",
                    to_number, row["id"], row["name"],
                )
                return str(row["id"])

            if to_number.startswith("sip:"):
                user_part = to_number.split("sip:")[1].split("@")[0]
                row = await conn.fetchrow(
                    "SELECT id, name FROM tenants WHERE twilio_number = $1", user_part
                )
                if row:
                    logger.info(
                        "tenant 매칭 (sip user=%s) to=%s → id=%s name=%s",
                        user_part, to_number, row["id"], row["name"],
                    )
                    return str(row["id"])
        finally:
            await conn.close()
    except Exception as e:
        logger.warning(f"tenant 조회 실패 to={to_number} err={e}")

    logger.warning(
        f"미등록 tenant to={to_number} — raw 값으로 진행 (DB에 번호 등록 필요)"
    )
    return to_number


async def get_tenant_name(tenant_id: str) -> str:
    """tenant_id 로 테넌트 표시명 조회. 미등록/오류 시 '고객센터' 반환.

    in-process 캐시 사용 — 통화당 1회 조회로 charge.
    """
    if tenant_id in _tenant_name_cache:
        return _tenant_name_cache[tenant_id]
    if not _UUID_RE.match(tenant_id):
        return "고객센터"
    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                "SELECT name FROM tenants WHERE id = $1::uuid", tenant_id
            )
            if row:
                _tenant_name_cache[tenant_id] = row["name"]
                return row["name"]
        finally:
            await conn.close()
    except Exception as e:
        logger.warning("tenant name 조회 실패 tenant_id=%s err=%s", tenant_id, e)
    return "고객센터"


async def get_greeting(tenant_id: str, within_hours: bool) -> str:
    """tenants.settings JSONB 에서 greeting 조회. 부재·오류 시 기본값 반환.

    `within_hours=False` 인데 `offhours_greeting` 이 미설정이면 일반 `greeting` 으로 폴백
    (업무시간 외 전화에도 병원/업체 맞춤 인사말 재사용).
    """
    field = "greeting" if within_hours else "offhours_greeting"
    default = DEFAULT_GREETING if within_hours else DEFAULT_OFFHOURS_GREETING

    if not _UUID_RE.match(tenant_id):
        logger.warning("get_greeting: 유효하지 않은 tenant_id=%s — 기본값 사용", tenant_id)
        return default

    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                "SELECT settings, name FROM tenants WHERE id = $1::uuid", tenant_id
            )
            if row:
                settings_data = row["settings"] or {}
                if isinstance(settings_data, str):
                    settings_data = _json.loads(settings_data)
                msg = settings_data.get(field)
                if msg:
                    return msg
                if not within_hours:
                    fallback = settings_data.get("greeting")
                    if fallback:
                        logger.info(
                            "offhours_greeting 미설정 → greeting 폴백 tenant_id=%s",
                            tenant_id,
                        )
                        return fallback
                # settings.greeting 미설정 시 tenant.name 으로 동적 인사말 생성
                tenant_name = row["name"]
                if tenant_name:
                    if within_hours:
                        return f"안녕하세요, {tenant_name}입니다. 무엇을 도와드릴까요?"
                    return (
                        f"안녕하세요, {tenant_name}입니다. "
                        f"현재 상담원 운영 시간이 아니지만 기본적인 문의는 도와드릴 수 있습니다. "
                        f"무엇을 도와드릴까요?"
                    )
        finally:
            await conn.close()
    except Exception as e:
        logger.warning("greeting 조회 실패 tenant_id=%s err=%s", tenant_id, e)
    return default
