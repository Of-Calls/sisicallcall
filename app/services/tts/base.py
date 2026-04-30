from abc import ABC, abstractmethod


class BaseTTSService(ABC):
    """저수준 TTS 합성. `BaseTTSOutputChannel` 내부에서 바이트를 얻어 WebSocket 으로 push 한다."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """텍스트를 Twilio Media Stream 호환 포맷(μ-law 8kHz mono)으로 합성."""
        raise NotImplementedError

    async def synthesize_and_stream(self, text: str) -> None:
        """(deprecated) 레거시 tts_node 호환용. 기본 구현은 synthesize() 만 호출."""
        await self.synthesize(text)


class BaseTTSOutputChannel(ABC):
    """오디오 출력 전용 파이프라인.

    lifecycle:
        open(call_id, tenant_id) → push_response → flush(call_id)

    barge-in:
        cancel(call_id) — current stream 중단
    """

    @abstractmethod
    async def open(self, call_id: str, tenant_id: str) -> None:
        """통화 시작. Twilio Media Stream 핸들 바인딩."""

    @abstractmethod
    async def push_response(self, call_id: str, text: str, response_path: str) -> None:
        """최종 응답 push."""

    @abstractmethod
    async def cancel(self, call_id: str) -> None:
        """barge-in 시 호출 — current stream 중단."""

    @abstractmethod
    async def flush(self, call_id: str) -> None:
        """통화 종료. per-call 내부 상태 정리."""

    def is_speaking(self, call_id: str) -> bool:
        """현재 call_id 의 응답이 TTS 송신 중이면 True. barge-in 감지에 사용."""
        return False

    def current_text(self, call_id: str) -> str:
        """현재 송신 중인 응답 원문. 없으면 빈 문자열."""
        return ""
