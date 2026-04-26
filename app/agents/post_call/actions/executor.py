from __future__ import annotations
from app.agents.post_call.schemas import Tool
from app.agents.post_call.actions.gmail_action import GmailAction
from app.agents.post_call.actions.company_db_action import CompanyDBAction
from app.agents.post_call.actions.calendar_action import CalendarAction
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TOOL_HANDLERS: dict[Tool, object] = {
    Tool.gmail: GmailAction(),
    Tool.company_db: CompanyDBAction(),
    Tool.calendar: CalendarAction(),
}


class ActionExecutor:
    async def execute_all(self, actions: list[dict], *, call_id: str) -> list[dict]:
        results = []
        for action in actions:
            results.append(await self._execute_one(action, call_id=call_id))
        return results

    async def _execute_one(self, action: dict, *, call_id: str) -> dict:
        tool_key = action.get("tool")
        try:
            tool_enum = Tool(tool_key)
        except ValueError:
            return {**action, "status": "skipped", "error": f"unknown tool: {tool_key}"}

        if tool_enum == Tool.internal_dashboard:
            return {**action, "status": "success", "result": {"note": "internal_dashboard 직접 처리"}}

        handler = _TOOL_HANDLERS.get(tool_enum)
        if handler is None:
            return {**action, "status": "skipped", "error": f"no handler for {tool_key}"}

        try:
            result = await handler.execute(action, call_id=call_id)  # type: ignore[attr-defined]
            return {**action, "status": "success", "result": result}
        except Exception as exc:
            logger.error(
                "action 실패 call_id=%s tool=%s action_type=%s err=%s",
                call_id, tool_key, action.get("action_type"), exc,
            )
            return {**action, "status": "failed", "error": str(exc)}
