"""Rolling PCM buffer — speculative 진입 직전 사용자 음성 보존.

verification 입력 prepend 용 (앞쪽 ~300ms 잘려나가는 것 방지).
linear16 16kHz 가정 (call.py audioop.ratecv 출력과 동일).

Phase 5 에서 Twilio mark 와 결합 — get_spoken_portion_via_mark.
"""
from app.barge_in.constants import ROLLING_BUFFER_MS, SAMPLE_RATE_HZ


class RollingPCMBuffer:
    def __init__(self, max_ms: int = ROLLING_BUFFER_MS) -> None:
        # linear16 = 2 bytes/sample
        self.max_bytes = int(max_ms * SAMPLE_RATE_HZ / 1000) * 2
        self.buf = bytearray()

    def append(self, pcm_16k: bytes) -> None:
        self.buf.extend(pcm_16k)
        if len(self.buf) > self.max_bytes:
            del self.buf[: len(self.buf) - self.max_bytes]

    def get_recent_ms(self, ms: int) -> bytes:
        n_bytes = int(ms * SAMPLE_RATE_HZ / 1000) * 2
        if n_bytes >= len(self.buf):
            return bytes(self.buf)
        return bytes(self.buf[-n_bytes:])

    def clear(self) -> None:
        self.buf.clear()
