import asyncio

from app.agents.conversational.state import CallState
from app.agents.summary.sync_mode import SyncSummaryAgent
from app.services.session.redis_session import RedisSessionService
from app.utils.logger import get_logger

# LLM 없음 — 운영시간 · 상담원 가용성 기반 결정적 분기 (~50ms)
# 3분할: immediate / callback / offhours (feature_spec.md §6)
# immediate 확정 시 Summary 동기 모드 직접 호출 (3초 하드컷)

logger = get_logger(__name__)

_session_service = RedisSessionService()
_sync_summary = SyncSummaryAgent()

SUMMARY_SYNC_TIMEOUT_SEC = 3.0

# TODO(agents.md 이관): tenant.settings JSONB 업종별 오버라이드 전까지 기본 멘트
MSG_IMMEDIATE = "상담원에게 즉시 연결해 드리겠습니다."
MSG_CALLBACK = "현재 모든 상담원이 통화 중이어서, 가능한 빠른 시간 내 콜백 드리겠습니다."
MSG_OFFHOURS = "현재는 운영 시간이 아닙니다. 운영 시간에 다시 연락 주시기 바랍니다."


async def escalation_branch_node(state: CallState) -> dict:
    tenant_id = state["tenant_id"]
    call_id = state["call_id"]

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

    # immediate — Summary 동기 모드 직접 호출 (이벤트 큐 경유 금지, feature_spec §6.6)
    try:
        await asyncio.wait_for(
            _sync_summary.run(call_id=call_id, tenant_id=tenant_id),
            timeout=SUMMARY_SYNC_TIMEOUT_SEC,
        )
        logger.info("escalation sub_state=immediate call_id=%s summary=ok", call_id)
    except asyncio.TimeoutError:
        # 타임아웃 시 handoff_notes 비운 채 통화 전환 — 상담원이 실시간 transcript 조회
        logger.warning("escalation immediate summary timeout call_id=%s", call_id)
    except Exception as e:
        logger.error("escalation immediate summary error call_id=%s: %s", call_id, e)

    return {
        "response_text": MSG_IMMEDIATE,
        "response_path": "escalation",
        "is_timeout": False,
    }
