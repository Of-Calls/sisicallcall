from typing import Literal, TypedDict

from app.services.tts.base import BaseTTSOutputChannel
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmittedEvent(TypedDict):
    type: Literal["response"]
    call_id: str
    text: str
    metadata: dict  # response: {response_path}


class MockTTSOutputChannel(BaseTTSOutputChannel):
    def __init__(self):
        self._open_calls: dict[str, str] = {}
        self._emissions: dict[str, list[EmittedEvent]] = {}
        self._cancelled: dict[str, bool] = {}
        self._is_speaking: dict[str, bool] = {}
        self._current_text: dict[str, str] = {}

    # ── lifecycle ─────────────────────────────────────────────

    async def open(self, call_id: str, tenant_id: str) -> None:
        self._open_calls[call_id] = tenant_id
        self._emissions[call_id] = []
        self._cancelled[call_id] = False
        self._is_speaking[call_id] = False
        self._current_text[call_id] = ""

    async def flush(self, call_id: str) -> None:
        self._open_calls.pop(call_id, None)
        self._cancelled.pop(call_id, None)
        self._is_speaking.pop(call_id, None)
        self._current_text.pop(call_id, None)

    # ── emit ──────────────────────────────────────────────────

    async def push_response(self, call_id: str, text: str, response_path: str) -> None:
        if call_id not in self._open_calls:
            logger.warning("push_response on un-opened call_id=%s — ignored", call_id)
            return
        self._current_text[call_id] = text
        logger.info("push_response call_id=%s path=%s text=%r", call_id, response_path, text[:200])
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
        self._emissions[call_id].clear()

    # ── 상태 조회 ────────────────────────────────────────────

    def is_speaking(self, call_id: str) -> bool:
        return self._is_speaking.get(call_id, False)

    def current_text(self, call_id: str) -> str:
        return self._current_text.get(call_id, "")

    # ── 테스트 검증용 헬퍼 ───────────────────────────────────

    def events_for(self, call_id: str) -> list[EmittedEvent]:
        return list(self._emissions.get(call_id, []))

    def is_cancelled(self, call_id: str) -> bool:
        return self._cancelled.get(call_id, False)

    def is_open(self, call_id: str) -> bool:
        return call_id in self._open_calls
