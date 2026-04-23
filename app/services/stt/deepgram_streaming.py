"""Deepgram Live (Streaming) STT — 통화당 WebSocket 1개 유지, 실시간 transcript 누적.

Prerecorded(batch) 대비 STT 레이턴시 2~3초 → ~300ms 단축.
발화 감지 침묵 구간(~800ms) 동안 Deepgram이 이미 is_final 결과를 반환하므로
flush_transcript() 호출 시 대부분 즉시 반환.

사용 순서:
    open(call_id)              ← 통화 시작 시 ("start" 이벤트)
    send(call_id, pcm_bytes)   ← 매 PCM 청크 도착 시 ("media" 이벤트)
    flush_transcript(call_id)  ← 발화 종료 감지 시 (silence trigger)
    close(call_id)             ← 통화 종료 시 ("stop" / disconnect)
"""
import asyncio
from dataclasses import dataclass, field

from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_LIVE_OPTIONS = LiveOptions(
    model="nova-3",
    language="ko",
    encoding="linear16",
    sample_rate=16000,
    smart_format=True,
    punctuate=True,
    interim_results=True,
    utterance_end_ms="1000",  # 1초 침묵 후 발화 종료 판단 — 어절 간 짧은 쉼 허용
    endpointing=300,           # 음절 사이 짧은 pause 에도 즉시 finalize 방지
)


@dataclass
class _CallStream:
    connection: object
    parts: list[str] = field(default_factory=list)


class DeepgramStreamingSTTService:
    """통화당 Deepgram 스트리밍 WebSocket 연결을 관리."""

    def __init__(self):
        self._client = DeepgramClient(settings.deepgram_api_key)
        self._streams: dict[str, _CallStream] = {}

    async def open(self, call_id: str) -> None:
        """통화 시작 시 Deepgram 스트리밍 연결 개설."""
        if call_id in self._streams:
            logger.warning("스트리밍 연결 중복 개설 call_id=%s — 기존 연결 재사용", call_id)
            return

        state = _CallStream(connection=None)
        connection = self._client.listen.asyncwebsocket.v("1")

        async def _on_transcript(self_dg, result, **kwargs):
            try:
                sentence = result.channel.alternatives[0].transcript
                if result.is_final and sentence.strip():
                    state.parts.append(sentence)
                    logger.debug("STT is_final call_id=%s text='%s'", call_id, sentence)
            except Exception as exc:
                logger.warning("STT 스트리밍 콜백 오류 call_id=%s: %s", call_id, exc)

        connection.on(LiveTranscriptionEvents.Transcript, _on_transcript)

        started = await connection.start(_LIVE_OPTIONS)
        if not started:
            raise RuntimeError(f"Deepgram 스트리밍 연결 실패 call_id={call_id}")

        state.connection = connection
        self._streams[call_id] = state
        logger.info("Deepgram 스트리밍 연결 개설 call_id=%s", call_id)

    async def send(self, call_id: str, pcm_chunk: bytes) -> None:
        """PCM 16kHz 16-bit 청크를 Deepgram 스트림에 전송."""
        stream = self._streams.get(call_id)
        if stream and stream.connection:
            try:
                await stream.connection.send(pcm_chunk)
            except Exception as exc:
                logger.warning("STT 스트리밍 send 오류 call_id=%s: %s", call_id, exc)

    async def flush_transcript(self, call_id: str) -> str:
        """발화 종료 시 누적 transcript 반환 후 버퍼 초기화.

        침묵 구간 800ms 동안 Deepgram이 is_final을 이미 반환하므로 대부분 즉시 반환.
        버퍼가 비었을 때만 100ms 대기 후 재확인.
        """
        stream = self._streams.get(call_id)
        if not stream:
            return ""

        if not stream.parts:
            await asyncio.sleep(0.1)

        transcript = " ".join(stream.parts).strip()
        stream.parts.clear()
        logger.info("STT flush call_id=%s result='%s'", call_id, transcript)
        return transcript

    async def close(self, call_id: str) -> None:
        """통화 종료 시 연결 정리."""
        stream = self._streams.pop(call_id, None)
        if stream and stream.connection:
            try:
                await stream.connection.finish()
            except Exception as exc:
                logger.warning("STT 스트리밍 종료 오류 call_id=%s: %s", call_id, exc)
        logger.info("Deepgram 스트리밍 연결 종료 call_id=%s", call_id)
