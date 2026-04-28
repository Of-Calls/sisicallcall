"""
Action Handler Registry.

새 MCP tool을 추가하는 방법:
1. app/agents/post_call/actions/{name}_action.py 파일 생성
2. execute(self, action, *, call_id, tenant_id="") -> dict 메서드 구현
   - 반환 형식: {"external_id": ..., "status": "success"|"failed", "result": {...}}
3. 이 파일 하단 _register_defaults() 안에 register("{name}", YourAction()) 추가
4. (선택) app/services/mcp/{name}.py 에 실제 연동 서비스 구현
"""
from __future__ import annotations

from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Registry 저장소 ───────────────────────────────────────────────────────────

_registry: dict[str, Any] = {}


def register(tool_name: str, handler: Any) -> None:
    """tool_name 에 handler 를 등록한다. 이미 존재하면 덮어쓴다."""
    _registry[tool_name] = handler
    logger.debug("registry: registered tool=%s handler=%s", tool_name, type(handler).__name__)


def unregister(tool_name: str) -> None:
    """tool_name 등록을 해제한다. 없어도 오류 없이 무시한다."""
    _registry.pop(tool_name, None)


def get_handler(tool_name: str) -> Any | None:
    """등록된 handler를 반환한다. 없으면 None."""
    return _registry.get(tool_name)


def registered_tools() -> list[str]:
    """현재 등록된 tool 이름 목록을 반환한다."""
    return list(_registry.keys())


# ── Internal Dashboard Handler ────────────────────────────────────────────────

class InternalDashboardAction:
    """내부 대시보드 전용 핸들러 — 외부 MCP 없이 즉시 성공 처리한다."""

    async def execute(
        self,
        action: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        logger.info(
            "InternalDashboardAction call_id=%s action_type=%s",
            call_id, action.get("action_type"),
        )
        return {
            "external_id": f"dashboard-{call_id}",
            "status": "success",
            "result": {"note": "internal_dashboard 직접 처리"},
        }


# ── Default registrations ────────────────────────────────────────────────────
# 기본 4개 tool을 모듈 로드 시점에 등록한다.
# 새 tool을 추가할 때는 이 함수만 수정하면 된다.

def _register_defaults() -> None:
    from app.agents.post_call.actions.gmail_action import GmailAction
    from app.agents.post_call.actions.company_db_action import CompanyDBAction
    from app.agents.post_call.actions.calendar_action import CalendarAction
    from app.agents.post_call.actions.jira_action import JiraAction
    from app.agents.post_call.actions.slack_action import SlackAction

    register("gmail", GmailAction())
    register("company_db", CompanyDBAction())
    register("calendar", CalendarAction())
    register("internal_dashboard", InternalDashboardAction())
    register("jira", JiraAction())
    register("slack", SlackAction())


_register_defaults()
