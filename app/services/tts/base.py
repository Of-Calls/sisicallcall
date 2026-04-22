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
    """오디오 출력 전용 파이프라인 (RFC 001 v0.2).

    노드가 stall utterance 와 최종 응답을 TTS/Twilio 로 라우팅할 때 사용한다.
    Ordering, cancel (barge-in), 사전 캐시 조회, 멀티테넌시 격리를 이 계층에 집약한다.

    lifecycle:
        open(call_id, tenant_id)
            → (push_stall)*  # stall 은 턴당 최대 1회 (이중 방출 방지)
            → push_response  # 최종 응답
            → flush(call_id)

    barge-in:
        cancel(call_id) — current stream 중단 + 큐 클리어

    실구현체는 `BaseTTSService` 를 내부 의존성으로 가진다 (합성/스트림 위임).
    """

    @abstractmethod
    async def open(self, call_id: str, tenant_id: str) -> None:
        """턴 시작. Twilio Media Stream 핸들 바인딩 + 내부 큐 초기화."""

    @abstractmethod
    async def push_stall(self, call_id: str, text: str, audio_field: str) -> None:
        """대기 멘트 push. 이미 같은 턴에 stall 이 방출됐으면 skip (이중 방출 방지).

        실구현체는 사전 캐시 (Redis `tenant:{tenant_id}:stall_audio_cache`) 를 우선
        조회하여 적중 시 TTS API 스킵. Miss 시 TTS 생성 후 write-back.
        """

    @abstractmethod
    async def push_response(self, call_id: str, text: str, response_path: str) -> None:
        """최종 응답 push. Reviewer 결과 반영된 최종 텍스트."""

    @abstractmethod
    async def cancel(self, call_id: str) -> None:
        """barge-in 시 호출 — current stream 중단 + 큐 클리어.

        `stall_emitted_this_turn` 플래그는 reset 하지 않음 (같은 턴 내 재진입 시 혼란 방지).
        플래그 reset 은 flush() 에서만.
        """

    @abstractmethod
    async def flush(self, call_id: str) -> None:
        """턴 종료. 내부 상태 (큐 + stall_emitted 플래그 + tenant mapping) 정리."""
