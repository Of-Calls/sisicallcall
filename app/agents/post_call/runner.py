"""
Post-call Agent 안전 실행 래퍼.

통화 종료 hook(Twilio stop 이벤트 등)에서 호출하기 위한 진입점.
내부 예외를 절대 밖으로 전파하지 않으므로 실시간 통화 응답을 blocking하지 않는다.

KDT-79 실제 연결 시 app/api/v1/call.py 의 stop 이벤트 핸들러에서
아래 중 하나의 방식으로 호출한다:
  asyncio.create_task(run_post_call_agent_safely(call_id, trigger, tenant_id))
  또는
  BackgroundTasks.add_task(run_post_call_agent_safely, call_id, trigger, tenant_id)
"""
from __future__ import annotations

from app.agents.post_call.agent import PostCallAgent
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def run_post_call_agent_safely(
    call_id: str,
    trigger: str = "call_ended",
    tenant_id: str = "default",
) -> dict:
    """Post-call Agent를 안전하게 실행한다.

    - 성공: {"ok": True,  "result": <PostCallAgentState dict>, "error": None}
    - 실패: {"ok": False, "result": None, "error": "<error message>"}

    예외는 절대 밖으로 전파하지 않는다.
    실패 시 logger.error/exception으로 기록한다.

    LLM 정책:
      POST_CALL_USE_REAL_LLM=true 이면 실제 GPT-4o를 사용한다.
      그 외에는 MockLLMCaller를 사용한다 (테스트 및 기본 실행 모두 안전).
    """
    logger.info(
        "run_post_call_agent_safely 시작 call_id=%s trigger=%s tenant_id=%s",
        call_id, trigger, tenant_id,
    )
    try:
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
