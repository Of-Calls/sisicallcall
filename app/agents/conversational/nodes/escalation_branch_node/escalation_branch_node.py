import asyncio

from app.agents.conversational.state import CallState
from app.agents.summary.sync_mode import SyncSummaryAgent
from app.services.session.redis_session import RedisSessionService
from app.services.tts.channel import tts_channel
from app.utils.logger import get_logger

# LLM 없음 — 운영시간 · 상담원 가용성 기반 결정적 분기 (~50ms)
# 3분할: immediate / callback / offhours (feature_spec.md §6)
# immediate 확정 시 Summary 동기 모드 직접 호출 (3초 하드컷, RFC 001 v0.2 §6.8 — stall 적용)

logger = get_logger(__name__)

_session_service = RedisSessionService()
_sync_summary = SyncSummaryAgent()

SUMMARY_SYNC_TIMEOUT_SEC = 3.0
STALL_DELAY_DEFAULT = 1.0

# TODO(agents.md 이관): tenant.settings JSONB 업종별 오버라이드 전까지 기본 멘트
MSG_IMMEDIATE = "상담원에게 즉시 연결해 드리겠습니다."
MSG_CALLBACK = "현재 모든 상담원이 통화 중이어서, 가능한 빠른 시간 내 콜백 드리겠습니다."
MSG_OFFHOURS = "현재는 운영 시간이 아닙니다. 운영 시간에 다시 연락 주시기 바랍니다."


def _pick_stall_msg(state: CallState) -> str:
    msgs = state.get("stall_messages") or {}
    return msgs.get("general") or "잠시만요, 확인해 드리겠습니다."


async def _run_summary_with_stall(
    *, call_id: str, tenant_id: str, stall_msg: str, stall_delay: float
) -> None:
    """Summary Sync 실행 + stall race. 응답은 MSG_IMMEDIATE 고정이라 리턴값 없음.

    Summary 결과는 Redis 에 저장되어 상담원 인수인계용으로 쓰이며, 본 턴의 TTS 응답과
    무관. 그러나 3초 대기 중 침묵 방지 위해 stall 규칙 동일 적용.
    """
    summary_task = asyncio.create_task(
        _sync_summary.run(call_id=call_id, tenant_id=tenant_id)
    )
    try:
        await asyncio.wait_for(asyncio.shield(summary_task), timeout=stall_delay)
        return  # 1초 안에 완료 — stall 불필요
    except asyncio.TimeoutError:
        pass
    except Exception as e:
        logger.error("escalation immediate summary phase1 error call_id=%s: %s", call_id, e)
        summary_task.cancel()
        return

    # stall 방출 후 남은 시간 대기
    await tts_channel.push_stall(
        call_id=call_id, text=stall_msg, audio_field="general"
    )
    remaining = max(SUMMARY_SYNC_TIMEOUT_SEC - stall_delay, 0.0)
    try:
        await asyncio.wait_for(summary_task, timeout=remaining)
    except asyncio.TimeoutError:
        logger.warning("escalation immediate summary timeout call_id=%s", call_id)
    except Exception as e:
        logger.error("escalation immediate summary phase2 error call_id=%s: %s", call_id, e)


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

    # immediate — Summary 동기 모드 + stall race
    await _run_summary_with_stall(
        call_id=call_id,
        tenant_id=tenant_id,
        stall_msg=_pick_stall_msg(state),
        stall_delay=state.get("stall_delay_sec", STALL_DELAY_DEFAULT),
    )
    logger.info("escalation sub_state=immediate call_id=%s", call_id)

    return {
        "response_text": MSG_IMMEDIATE,
        "response_path": "escalation",
        "is_timeout": False,
    }
