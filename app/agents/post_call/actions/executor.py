from __future__ import annotations

from app.agents.post_call.actions.registry import get_handler
from app.agents.post_call.actions.result import action_failed, action_skipped, action_success
from app.repositories.mcp_action_log_repo import find_successful_action
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ActionExecutor:
    """action_plan.actions 를 registry 에서 찾은 handler 로 라우팅하고
    표준 6-key 결과 list 를 반환한다.

    새 tool 추가 시 executor.py 는 수정하지 않는다.
    registry.py 에 handler 만 등록하면 된다.
    """

    async def execute_actions(
        self,
        call_id: str,
        tenant_id: str,
        actions: list[dict] | None,
    ) -> list[dict]:
        """표준 인터페이스.

        - actions 가 None 이거나 빈 list 면 [] 반환.
        - 하나의 action 이 실패해도 나머지 action 은 계속 실행한다.
        - 반환 순서는 입력 actions 순서와 동일하다.
        """
        if not actions:
            return []
        results: list[dict] = []
        for action in actions:
            results.append(
                await self._execute_one(action, call_id=call_id, tenant_id=tenant_id)
            )
        return results

    async def execute_all(self, actions: list[dict], *, call_id: str) -> list[dict]:
        """후방 호환 인터페이스 — action_router_node 가 호출한다."""
        return await self.execute_actions(
            call_id=call_id,
            tenant_id="",
            actions=actions,
        )

    async def _execute_one(
        self,
        action: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        tool_key = action.get("tool", "")
        action_type = action.get("action_type", "")

        handler = get_handler(tool_key)
        if handler is None:
            logger.warning(
                "알 수 없는 tool call_id=%s tool=%r action_type=%s",
                call_id, tool_key, action_type,
            )
            return action_failed(action, error=f"unknown tool: {tool_key!r}")

        previous = await find_successful_action(
            call_id=call_id,
            action_type=action_type,
            tool=tool_key,
        )
        if previous:
            logger.info(
                "action idempotency skip call_id=%s tool=%s action_type=%s previous_external_id=%s",
                call_id,
                tool_key,
                action_type,
                previous.get("external_id"),
            )
            return action_skipped(
                action,
                reason="already_succeeded",
                result={
                    "idempotency": "already_succeeded",
                    "previous_external_id": previous.get("external_id"),
                    "previous_status": previous.get("status"),
                },
            )

        try:
            raw: dict = await handler.execute(
                action,
                call_id=call_id,
                tenant_id=tenant_id,
            )
            status = raw.get("status", "success")
            if status == "failed":
                return action_failed(
                    action,
                    error=raw.get("error") or "handler returned failed",
                    result=raw.get("result"),
                )
            if status == "skipped":
                return action_skipped(
                    action,
                    reason=raw.get("error") or "handler returned skipped",
                    result=raw.get("result"),
                )
            return action_success(
                action,
                external_id=raw.get("external_id"),
                result=raw.get("result"),
            )
        except Exception as exc:
            logger.error(
                "action 실패 call_id=%s tool=%s action_type=%s err=%s",
                call_id, tool_key, action_type, exc,
            )
            return action_failed(action, error=str(exc))


# ── 모듈 레벨 편의 함수 ───────────────────────────────────────────────────────

_default_executor = ActionExecutor()


async def execute_actions(
    call_id: str,
    tenant_id: str,
    actions: list[dict] | None,
) -> list[dict]:
    """모듈 레벨 편의 함수 — ActionExecutor().execute_actions() 와 동일."""
    return await _default_executor.execute_actions(
        call_id=call_id,
        tenant_id=tenant_id,
        actions=actions,
    )
