from __future__ import annotations
from app.agents.post_call.graph import build_post_call_graph
from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_TRIGGERS = frozenset({"call_ended", "escalation_immediate", "manual"})


class PostCallAgent:
    def __init__(self) -> None:
        self._graph = build_post_call_graph()

    async def run(
        self,
        call_id: str,
        trigger: str,
        tenant_id: str = "default",
    ) -> PostCallAgentState:
        """통화 후처리 에이전트 실행 엔트리포인트.

        trigger:
          - call_ended           : 정상 통화 완료 → 풀 파이프라인
          - escalation_immediate : 즉시 에스컬레이션 → summary만 실행
          - manual               : 수동 재처리 → 풀 파이프라인
        """
        if trigger not in _VALID_TRIGGERS:
            raise ValueError(f"Unknown trigger: {trigger!r}. 허용값: {sorted(_VALID_TRIGGERS)}")

        initial: PostCallAgentState = {
            "call_id": call_id,
            "tenant_id": tenant_id,
            "trigger": trigger,
            "call_metadata": {},
            "transcripts": [],
            "branch_stats": {},
            "summary": None,
            "voc_analysis": None,
            "priority_result": None,
            "action_plan": None,
            "executed_actions": [],
            "dashboard_payload": None,
            "errors": [],
            "partial_success": False,
            # ── Review Gate 초기값 ─────────────────────────────────────────
            "analysis_result": None,
            "review_result": None,
            "review_verdict": None,
            "review_retry_count": 0,
            "human_review_required": False,
            "blocked_actions": [],
        }

        logger.info("PostCallAgent 시작 call_id=%s trigger=%s", call_id, trigger)
        result: PostCallAgentState = await self._graph.ainvoke(initial)
        logger.info(
            "PostCallAgent 완료 call_id=%s partial=%s errors=%d",
            call_id,
            result.get("partial_success"),  # type: ignore[call-overload]
            len(result.get("errors", [])),  # type: ignore[call-overload]
        )
        return result
