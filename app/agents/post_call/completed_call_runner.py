"""
종료된 통화 데이터 기반 후처리 실행 진입점.

run_post_call_agent_safely (runner.py) 는 call.py stop 이벤트 hook 전용이다.
종료 데이터 명시 실행(API 수동 트리거, 재처리 등)은 이 모듈을 사용한다.

차이점:
  run_post_call_agent_safely  — context 없어도 PostCallAgent empty fallback 실행
  run_post_call_for_completed_call — context 없으면 즉시 ok=False 반환
"""
from __future__ import annotations

from app.agents.post_call.agent import PostCallAgent
from app.agents.post_call.context_provider import get_call_context_for_post_call
from app.repositories.call_summary_repo import seed_call_context
from app.utils.logger import get_logger

logger = get_logger(__name__)

_CALL_CONTEXT_NOT_FOUND = "call_context_not_found"


async def run_post_call_for_completed_call(
    call_id: str,
    tenant_id: str = "default",
    trigger: str = "call_ended",
) -> dict:
    """종료된 통화 데이터를 기반으로 후처리 파이프라인을 실행한다.

    반환값:
      - context 없음: {"ok": False, "result": None, "error": "call_context_not_found"}
      - 성공:         {"ok": True,  "result": <PostCallAgentState dict>, "error": None}
      - runner 예외:  {"ok": False, "result": None, "error": "<error message>"}

    PostCallAgent partial_success=True 결과는 ok=True로 반환한다.
    ok=False는 context를 찾지 못했거나 runner 자체가 예외로 실패한 경우에만 반환한다.
    """
    logger.info(
        "run_post_call_for_completed_call 시작 call_id=%s trigger=%s tenant_id=%s",
        call_id, trigger, tenant_id,
    )
    try:
        # ── Step 1: context 조회 ───────────────────────────────────────────────
        ctx = await get_call_context_for_post_call(call_id, tenant_id=tenant_id)

        if ctx is None:
            logger.warning(
                "run_post_call_for_completed_call: context 없음 call_id=%s — 실행 중단",
                call_id,
            )
            return {"ok": False, "result": None, "error": _CALL_CONTEXT_NOT_FOUND}

        # ── Step 2: repository에 seed ─────────────────────────────────────────
        await seed_call_context(
            call_id=call_id,
            tenant_id=tenant_id,
            transcripts=ctx.get("transcripts"),
            call_metadata=ctx.get("metadata"),
            branch_stats=ctx.get("branch_stats"),
        )
        logger.info(
            "context seed 완료 call_id=%s transcripts=%d",
            call_id, len(ctx.get("transcripts") or []),
        )

        # ── Step 3: PostCallAgent 실행 ────────────────────────────────────────
        agent = PostCallAgent()
        result = await agent.run(
            call_id=call_id,
            trigger=trigger,
            tenant_id=tenant_id,
        )

        partial = result.get("partial_success", False)  # type: ignore[call-overload]
        errors  = result.get("errors", [])              # type: ignore[call-overload]
        logger.info(
            "run_post_call_for_completed_call 완료 call_id=%s partial=%s errors=%d",
            call_id, partial, len(errors),
        )
        return {"ok": True, "result": dict(result), "error": None}

    except Exception as exc:
        logger.exception(
            "run_post_call_for_completed_call 실패 call_id=%s err=%s",
            call_id, exc,
        )
        return {"ok": False, "result": None, "error": str(exc)}
