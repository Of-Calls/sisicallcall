"""
Post-call Agent 안전 실행 래퍼.

통화 종료 hook(Twilio stop 이벤트)에서 호출하기 위한 진입점.
내부 예외를 절대 밖으로 전파하지 않으므로 실시간 통화 응답을 blocking하지 않는다.

실행 흐름:
  1. context_provider로 실제 통화 context 조회 (DB→Redis→in-memory→None 순)
  2. context가 있으면 repository에 seed → load_context_node가 사용
  3. context가 없어도 PostCallAgent는 empty fallback으로 실행
  4. PostCallAgent가 partial_success=True로 끝나도 ok=True 반환
  5. runner 자체가 예외로 실패한 경우에만 ok=False 반환

호출 방법 (app/api/v1/call.py stop 이벤트):
  asyncio.create_task(run_post_call_agent_safely(call_id, "call_ended", tenant_id))
"""
from __future__ import annotations

from app.agents.post_call.agent import PostCallAgent
from app.agents.post_call.context_provider import get_call_context_for_post_call
from app.repositories.call_summary_repo import seed_call_context
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def run_post_call_agent_safely(
    call_id: str,
    trigger: str = "call_ended",
    tenant_id: str = "default",
) -> dict:
    """Post-call Agent를 안전하게 실행한다.

    반환값:
      - 성공: {"ok": True,  "result": <PostCallAgentState dict>, "error": None}
      - 실패: {"ok": False, "result": None, "error": "<error message>"}

    ok=False는 runner 내부가 예외로 실패한 경우에만 반환한다.
    PostCallAgent partial_success=True 결과는 ok=True로 반환한다.
    예외는 절대 밖으로 전파하지 않는다.

    LLM 정책:
      POST_CALL_USE_REAL_LLM=true 이면 실제 GPT-4o를 사용한다.
      그 외에는 MockLLMCaller를 사용한다 (테스트 및 기본 실행 모두 안전).
    """
    logger.info(
        "run_post_call_agent_safely 시작 call_id=%s trigger=%s tenant_id=%s",
        call_id, trigger, tenant_id,
    )
    try:
        # ── Step 1: 실제 통화 context 조회 ────────────────────────────────────
        # DB→Redis→in-memory→None 순으로 조회 (context_provider 우선순위)
        ctx = await get_call_context_for_post_call(call_id, tenant_id=tenant_id)

        # ── Step 2: context를 repository에 seed ───────────────────────────────
        # load_context_node가 seed된 데이터를 읽어 파이프라인에 주입한다.
        if ctx is not None:
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
        else:
            # context 없음 — PostCallAgent의 empty fallback 흐름으로 진행
            logger.info(
                "context 없음 call_id=%s — PostCallAgent empty fallback 진행",
                call_id,
            )

        # ── Step 3: PostCallAgent 실행 ─────────────────────────────────────────
        agent = PostCallAgent()
        result = await agent.run(
            call_id=call_id,
            trigger=trigger,
            tenant_id=tenant_id,
        )

        partial = result.get("partial_success", False)  # type: ignore[call-overload]
        errors = result.get("errors", [])               # type: ignore[call-overload]
        logger.info(
            "run_post_call_agent_safely 완료 call_id=%s partial=%s errors=%d",
            call_id, partial, len(errors),
        )
        return {"ok": True, "result": dict(result), "error": None}

    except Exception as exc:
        logger.exception(
            "run_post_call_agent_safely 실패 call_id=%s trigger=%s err=%s",
            call_id, trigger, exc,
        )
        return {"ok": False, "result": None, "error": str(exc)}
