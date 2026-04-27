from typing import Literal, TypedDict

from app.services.tts.base import BaseTTSOutputChannel
from app.utils.logger import get_logger

# 테스트 및 dev 환경 전용 Mock. Twilio / 실제 TTS API 호출 없이 내부 리스트에
# emit 이벤트를 기록만 한다. 테스트는 이 리스트를 assert 로 검증.

logger = get_logger(__name__)


class EmittedEvent(TypedDict):
    type: Literal["stall", "response"]
    call_id: str
    text: str
    metadata: dict  # stall: {audio_field}, response: {response_path}


class MockTTSOutputChannel(BaseTTSOutputChannel):
    def __init__(self):
        self._open_calls: dict[str, str] = {}          # call_id -> tenant_id
        self._stall_emitted: dict[str, bool] = {}       # call_id -> 이중 방출 방지
        self._emissions: dict[str, list[EmittedEvent]] = {}  # call_id -> emit 로그
        self._cancelled: dict[str, bool] = {}           # call_id -> cancel 플래그
        # barge-in 인터페이스 일치 — 송신 시뮬레이션은 없지만
        # current_text 는 직전 push_response 텍스트를 보존해 끊긴 응답 컨텍스트 추적 가능.
        self._is_speaking: dict[str, bool] = {}
        self._current_text: dict[str, str] = {}

    # ── lifecycle ─────────────────────────────────────────────

    async def open(self, call_id: str, tenant_id: str) -> None:
        self._open_calls[call_id] = tenant_id
        self._stall_emitted[call_id] = False
        self._emissions[call_id] = []
        self._cancelled[call_id] = False
        self._is_speaking[call_id] = False
        self._current_text[call_id] = ""

    async def flush(self, call_id: str) -> None:
        # 턴 종료 — 모든 per-call 상태 cleanup
        self._open_calls.pop(call_id, None)
        self._stall_emitted.pop(call_id, None)
        # _emissions 는 테스트 후 검증용이라 남김 — assert 용도
        self._cancelled.pop(call_id, None)
        self._is_speaking.pop(call_id, None)
        self._current_text.pop(call_id, None)

    # ── emit ──────────────────────────────────────────────────

    async def push_stall(self, call_id: str, text: str, audio_field: str) -> None:
        if call_id not in self._open_calls:
            logger.warning("push_stall on un-opened call_id=%s — ignored", call_id)
            return
        if self._stall_emitted.get(call_id):
            logger.debug("stall already emitted for call_id=%s — skip", call_id)
            return
        self._stall_emitted[call_id] = True
        self._emissions[call_id].append({
            "type": "stall",
            "call_id": call_id,
            "text": text,
            "metadata": {"audio_field": audio_field},
        })

    async def push_response(self, call_id: str, text: str, response_path: str) -> None:
        if call_id not in self._open_calls:
            logger.warning("push_response on un-opened call_id=%s — ignored", call_id)
            return
        # 송신 시뮬레이션 없음 — current_text 만 갱신해 barge-in 컨텍스트 노출.
        self._current_text[call_id] = text
        self._emissions[call_id].append({
            "type": "response",
            "call_id": call_id,
            "text": text,
            "metadata": {"response_path": response_path},
        })

    async def cancel(self, call_id: str) -> None:
        if call_id not in self._open_calls:
            logger.warning("cancel on un-opened call_id=%s — ignored", call_id)
            return
        self._cancelled[call_id] = True
        self._is_speaking[call_id] = False
        self._current_text[call_id] = ""
        # RFC 규약: current stream 중단 + 큐 클리어.
        # stall_emitted 플래그는 reset 하지 않음.
        self._emissions[call_id].clear()

    # ── 상태 조회 (barge-in 지원) ─────────────────────────────

    def is_speaking(self, call_id: str) -> bool:
        return self._is_speaking.get(call_id, False)

    def current_text(self, call_id: str) -> str:
        return self._current_text.get(call_id, "")

    # ── 테스트 검증용 헬퍼 ────────────────────────────────────

    def events_for(self, call_id: str) -> list[EmittedEvent]:
        return list(self._emissions.get(call_id, []))

    def stall_emitted_for(self, call_id: str) -> bool:
        return self._stall_emitted.get(call_id, False)

    def is_cancelled(self, call_id: str) -> bool:
        return self._cancelled.get(call_id, False)

    def is_open(self, call_id: str) -> bool:
        return call_id in self._open_calls
