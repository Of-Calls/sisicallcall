from __future__ import annotations

from typing import Any

from app.agents.post_call.actions.gmail_action import GmailAction
from app.agents.post_call.actions.company_db_action import CompanyDBAction
from app.agents.post_call.actions.calendar_action import CalendarAction
from app.utils.logger import get_logger

logger = get_logger(__name__)

# tool 문자열 → handler 인스턴스 (internal_dashboard 는 executor 내부 처리)
_TOOL_HANDLERS: dict[str, Any] = {
    "gmail": GmailAction(),
    "company_db": CompanyDBAction(),
    "calendar": CalendarAction(),
}


def _make_result(
    action: dict,
    *,
    status: str,
    external_id: str | None = None,
    error: str | None = None,
    result: dict | None = None,
) -> dict:
    """표준 6-key 결과를 생성한다. 원본 action 필드(params 등)도 보존한다."""
    return {
        **action,                               # action_type, tool, priority, params … 보존
        "action_type": action.get("action_type", ""),
        "tool": action.get("tool", ""),
        "status": status,
        "external_id": external_id,
        "error": error,
        "result": result if result is not None else {},
    }


class ActionExecutor:
    """action_plan.actions 를 tool 별 handler 로 라우팅하고 표준 결과 list 를 반환한다."""

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

        # internal_dashboard: 외부 핸들러 없이 executor 내부에서 직접 처리
        if tool_key == "internal_dashboard":
            logger.info(
                "internal_dashboard action call_id=%s action_type=%s",
                call_id, action_type,
            )
            return _make_result(
                action,
                status="success",
                external_id=f"dashboard-{call_id}",
                result={"note": "internal_dashboard 직접 처리"},
            )

        handler = _TOOL_HANDLERS.get(tool_key)
        if handler is None:
            logger.warning(
                "알 수 없는 tool call_id=%s tool=%r action_type=%s",
                call_id, tool_key, action_type,
            )
            return _make_result(
                action,
                status="failed",
                error=f"unknown tool: {tool_key!r}",
            )

        try:
            handler_result: dict = await handler.execute(  # type: ignore[attr-defined]
                action,
                call_id=call_id,
                tenant_id=tenant_id,
            )
            return _make_result(
                action,
                status=handler_result.get("status", "success"),
                external_id=handler_result.get("external_id"),
                result=handler_result.get("result", {}),
            )
        except Exception as exc:
            logger.error(
                "action 실패 call_id=%s tool=%s action_type=%s err=%s",
                call_id, tool_key, action_type, exc,
            )
            return _make_result(
                action,
                status="failed",
                error=str(exc),
            )


# ── 모듈 레벨 편의 함수 ───────────────────────────────────────────────────────
# ActionExecutor 를 직접 인스턴스화하지 않아도 되는 경우에 사용한다.

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
