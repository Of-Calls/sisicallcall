"""CallSession — 통화 1건의 라이프사이클 상태 캡슐화.

call_websocket() 의 nonlocal 변수 + inner function 들을 단일 객체로 정리.
teardown() 으로 stop / WebSocketDisconnect / Exception 종료 경로를 단일화.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from fastapi import WebSocket

from app.utils.logger import get_logger

logger = get_logger(__name__)

# task reference 보존 set — asyncio.create_task GC 방지 (Python 3.10+ 권고)
_background_tasks: set[asyncio.Task] = set()


def spawn_background(coro) -> asyncio.Task:
    """코루틴을 background task 로 등록. GC 방지용 set 에 보관."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


class CallSession:
    """통화 1건의 가변 상태 + 공통 라이프사이클 메서드."""

    __slots__ = (
        "websocket", "call_id", "tenant_id",
        "db_call_id", "call_started_at_monotonic",
        "pcm_buffer", "vad_window", "in_speech", "silence_chunk_count",
        "bargein_verify_attempted", "interrupted_response_text",
        "turn_index", "turn_task",
        "empty_stt_count", "silence_alert_count", "last_activity_at",
        "session_view", "channel_opened",
    )

    def __init__(self, websocket: WebSocket, call_id: str, tenant_id: str) -> None:
        self.websocket = websocket
        self.call_id = call_id
        self.tenant_id = tenant_id

        # DB 기록
        self.db_call_id: Optional[str] = None
        self.call_started_at_monotonic: Optional[float] = None

        # 발화 버퍼
        self.pcm_buffer = bytearray()
        self.vad_window = bytearray()   # Silero 슬라이딩 윈도우
        self.in_speech: bool = False
        self.silence_chunk_count: int = 0
        self.bargein_verify_attempted: bool = False
        self.interrupted_response_text: str = ""

        # turn 상태
        self.turn_index: int = 0
        self.turn_task: Optional[asyncio.Task] = None

        # 카운터 / 타이머
        self.empty_stt_count: int = 0
        self.silence_alert_count: int = 0
        self.last_activity_at: float = time.monotonic()

        # session_view — Intent Router / 노드들이 참조
        self.session_view: dict = {
            "tenant_name": "고객센터",
            "is_within_hours": True,
            "turn_count": 0,
            "last_intent": None,
            "last_question": None,
            "last_assistant_text": None,
            "clarify_count": 0,
            "rag_miss_count": 0,
            "auth_pending": False,
        }

        self.channel_opened: bool = False

    async def cancel_turn_task(self) -> None:
        """진행 중 turn task 취소."""
        if self.turn_task is not None and not self.turn_task.done():
            self.turn_task.cancel()
            try:
                await self.turn_task
            except (asyncio.CancelledError, Exception):
                pass
        self.turn_task = None

    async def teardown(self, status: str) -> None:
        """통화 종료 공통 cleanup.

        stop / WebSocketDisconnect / Exception 모든 종료 경로에서 호출.
        _streaming_stt.close() 는 call.py 에서 teardown() 직전에 호출.
        post-call 트리거는 모든 종료 경로에서 실행 (db_call_id 있을 때).
        """
        from app.agents.post_call.trigger import run_completed_post_call_background
        from app.repositories.call_repo import finalize_call
        from app.services.speaker_verify import enrollment as voiceprint_enrollment
        from app.services.speaker_verify.titanet import get_titanet_service
        from app.services.tts.channel import tts_channel
        from app.utils.audio import reset_resample_state

        await self.cancel_turn_task()
        reset_resample_state(self.call_id)

        if self.channel_opened:
            try:
                await tts_channel.flush(self.call_id)
            except Exception as exc:
                logger.warning("call_id=%s TTS flush 실패: %s", self.call_id, exc)

        if self.db_call_id:
            _duration = (
                int(time.monotonic() - self.call_started_at_monotonic)
                if self.call_started_at_monotonic else None
            )
            try:
                await finalize_call(
                    self.db_call_id, status=status, duration_sec=_duration
                )
            except Exception as exc:
                logger.error("call_id=%s finalize_call 실패: %s", self.call_id, exc)

            # post-call: 모든 종료 경로에서 실행
            spawn_background(
                run_completed_post_call_background(
                    call_id=self.call_id,
                    tenant_id=self.tenant_id,
                )
            )
        else:
            logger.warning(
                "post-call skip call_id=%s reason=db_call_id_missing", self.call_id
            )

        try:
            voiceprint_enrollment.cleanup(self.call_id)
            get_titanet_service().cleanup(self.call_id)
        except Exception as exc:
            logger.warning("call_id=%s speaker cleanup 실패: %s", self.call_id, exc)
