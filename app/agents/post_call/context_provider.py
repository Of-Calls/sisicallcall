"""
Post-call Context Provider.

통화 종료 후 후처리 에이전트가 사용할 통화 컨텍스트(transcript / metadata / branch_stats)를
여러 소스에서 우선순위 순으로 조회한다.

── 최종 운영 조회 순서 ───────────────────────────────────────────────────────
1. PostgreSQL  — calls / transcripts 테이블에서 call_id 기준 조회
   (실제 통화 데이터가 저장된 주 DB)
2. Redis       — session transcript fallback
   (통화 직후 DB 쓰기가 완료되지 않은 경우 Redis에서 조회)
3. Repository  — in-memory seed/mock (테스트 전용)
   (테스트 환경에서 seed_call_context로 사전 주입한 데이터)
4. None 반환   — 어디서도 찾지 못한 경우
   (load_context_node는 None을 받으면 fallback 처리)

── 반환 형식 ─────────────────────────────────────────────────────────────────
load_context_node가 기대하는 구조:
  {
    "metadata":     {"call_id": ..., "tenant_id": ..., ...},
    "transcripts":  [{"role": "customer"|"agent", "text": ...}, ...],
    "branch_stats": {"faq": int, "task": int, "escalation": int}
  }

── 현재 구현 ─────────────────────────────────────────────────────────────────
현재는 Step 3(repository in-memory)만 구현되어 있다.
Step 1 · 2는 KDT-79에서 실제 DB/Redis 연결 후 채운다.

── 주의 ──────────────────────────────────────────────────────────────────────
이 파일을 수정해 Step 1/2를 구현할 때:
- app/api/v1/call.py, app/main.py, 기존 통화 프로세스 코드를 수정하지 않는다.
- load_context_node.py는 이 함수를 직접 호출하지 않는다.
  KDT-79에서 load_context_node.py를 수정하거나 이 함수를 runner.py에서 호출한 뒤
  seed_call_context로 repository에 주입하는 방식을 선택한다.
"""
from __future__ import annotations

import copy

from app.repositories import get_seeded_call_context, seed_call_context
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def get_call_context_for_post_call(
    call_id: str,
    tenant_id: str | None = None,
) -> dict | None:
    """Post-call 에이전트가 사용할 통화 컨텍스트를 반환한다.

    조회 우선순위:
      1) TODO(KDT-79): PostgreSQL calls/transcripts 테이블 조회
      2) TODO(KDT-79): Redis session transcript fallback
      3) in-memory repository (seed_call_context로 주입된 데이터)
      4) None 반환

    반환값:
      {"metadata": {...}, "transcripts": [...], "branch_stats": {...}}
      또는 None (어디서도 찾지 못한 경우)

    deepcopy를 사용하므로 반환된 dict를 수정해도 내부 저장소가 오염되지 않는다.
    """
    # ── Step 1: PostgreSQL 조회 ────────────────────────────────────────────────
    # TODO(KDT-79): 아래 블록의 주석을 해제하고 실제 DB 서비스를 연결한다.
    #
    # from app.services.db.transcripts import get_transcripts_from_db
    # try:
    #     rows = await get_transcripts_from_db(call_id)
    #     if rows:
    #         metadata = await get_call_metadata_from_db(call_id)
    #         return {
    #             "metadata": {**metadata, "call_id": call_id, "tenant_id": tenant_id or ""},
    #             "transcripts": rows,
    #             "branch_stats": {},
    #         }
    # except Exception as exc:
    #     logger.warning("PostgreSQL 조회 실패 call_id=%s err=%s — fallback", call_id, exc)

    # ── Step 2: Redis session fallback ────────────────────────────────────────
    # TODO(KDT-79): 아래 블록의 주석을 해제하고 실제 Redis 서비스를 연결한다.
    #
    # from app.services.session.redis_session import RedisSessionService
    # try:
    #     session_svc = RedisSessionService()
    #     transcripts = await session_svc.get_transcripts(call_id)
    #     if transcripts:
    #         return {
    #             "metadata": {"call_id": call_id, "tenant_id": tenant_id or ""},
    #             "transcripts": transcripts,
    #             "branch_stats": {},
    #         }
    # except Exception as exc:
    #     logger.warning("Redis 조회 실패 call_id=%s err=%s — fallback", call_id, exc)

    # ── Step 3: in-memory repository (테스트 · 개발 환경) ────────────────────
    # get_seeded_call_context: sample fallback 없이 명시적으로 seed된 데이터만 반환
    ctx = await get_seeded_call_context(call_id)
    if ctx is not None:
        raw = copy.deepcopy(ctx)
        # repository의 get_seeded_call_context는 metadata/transcripts/branch_stats 구조를 반환한다.
        # load_context_node가 기대하는 형식과 동일하다.
        if raw.get("transcripts") or raw.get("metadata"):
            logger.debug("context_provider: repository fallback 사용 call_id=%s", call_id)
            return raw

    # ── Step 4: 찾지 못한 경우 ────────────────────────────────────────────────
    logger.warning(
        "context_provider: call_id=%s 에 대한 컨텍스트를 찾지 못함 — None 반환",
        call_id,
    )
    return None


async def seed_test_context(
    call_id: str,
    tenant_id: str = "default",
    transcripts: list[dict] | None = None,
    call_metadata: dict | None = None,
    branch_stats: dict | None = None,
) -> None:
    """테스트용 컨텍스트를 in-memory repository에 주입한다.

    운영 환경에서는 사용하지 않는다.
    테스트 코드에서 get_call_context_for_post_call을 검증할 때 호출한다.
    """
    await seed_call_context(
        call_id=call_id,
        tenant_id=tenant_id,
        transcripts=transcripts,
        call_metadata=call_metadata,
        branch_stats=branch_stats,
    )
    logger.debug("context_provider: test context seeded call_id=%s", call_id)
