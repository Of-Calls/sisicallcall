import os

from app.agents.conversational.state import CallState
from app.services.session.redis_session import RedisSessionService
from app.utils.logger import get_logger

# LLM 없음 — 운영시간 · 상담원 가용성 기반 결정적 분기 (~50ms)
# 3분할: immediate / callback / offhours (feature_spec.md §6)
#
# 2026-04-24: SyncSummaryAgent 호출 임시 제거. 후처리팀이 PostCallAgent.run_sync_summary()
# 메서드를 노출하면 그 시점에 immediate 분기에서 재연결. 자세한 결정은 CLAUDE.md 참조.

logger = get_logger(__name__)

_session_service = RedisSessionService()

# 영업시간 외 테스트 우회 토글 — .env 또는 셸에서 BYPASS_OFFHOURS=true 설정 시 항상 영업중 처리.
# 운영에선 미설정 → 정상 영업시간 검사 수행.
_BYPASS_OFFHOURS = os.getenv("BYPASS_OFFHOURS", "false").lower() == "true"

# TODO(agents.md 이관): tenant.settings JSONB 업종별 오버라이드 전까지 기본 멘트
MSG_IMMEDIATE = "상담원에게 즉시 연결해 드리겠습니다."
MSG_CALLBACK = "현재 모든 상담원이 통화 중이어서, 가능한 빠른 시간 내 콜백 드리겠습니다."
MSG_OFFHOURS = "현재 상담원 연결이 어렵습니다. 운영 시간에 콜백 드리겠습니다."


async def escalation_branch_node(state: CallState) -> dict:
    tenant_id = state["tenant_id"]
    call_id = state["call_id"]

    if _BYPASS_OFFHOURS:
        within_hours = True
    else:
        within_hours = await _session_service.is_within_business_hours(tenant_id)
    if not within_hours:
        logger.info("escalation sub_state=offhours call_id=%s", call_id)
        return {
            "response_text": MSG_OFFHOURS,
            "response_path": "escalation",
            "is_timeout": False,
        }

    agent_count = await _session_service.get_available_agent_count(tenant_id)
    if agent_count <= 0:
        logger.info("escalation sub_state=callback call_id=%s", call_id)
        return {
            "response_text": MSG_CALLBACK,
            "response_path": "escalation",
            "is_timeout": False,
        }

    logger.info("escalation sub_state=immediate call_id=%s", call_id)
    return {
        "response_text": MSG_IMMEDIATE,
        "response_path": "escalation",
        "is_timeout": False,
    }
