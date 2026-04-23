import asyncio
import audioop

from app.services.vad.base import BaseVADService
from app.utils.config import settings


def _energy_fallback(
    pcm16_16k: bytes, threshold: int = settings.webrtc_energy_fallback_threshold
) -> bool:
    if not pcm16_16k:
        return False
    return audioop.rms(pcm16_16k, 2) >= threshold


class WebRTCVADService(BaseVADService):
    name = "webrtc_vad"

    def __init__(self) -> None:
        self.available = True
        self.note = ""
        self._initialized = False
        self._vad = None

    async def initialize(self) -> None:
        if self._initialized:
            return
        try:
            import webrtcvad

            self._vad = webrtcvad.Vad(settings.webrtc_mode)
            self.available = True
            self.note = ""
        except Exception:
            self.available = False
            self.note = "webrtcvad 미설치, energy fallback 사용"
        finally:
            self._initialized = True

    async def detect(self, audio_chunk: bytes) -> bool:
        if not self._initialized:
            await self.initialize()

        def _infer() -> bool:
            if not self._vad:
                return _energy_fallback(audio_chunk)

            frame_size = int(16000 * settings.webrtc_frame_ms / 1000) * 2
            if len(audio_chunk) < frame_size:
                padded = audio_chunk + b"\x00" * (frame_size - len(audio_chunk))
                return self._vad.is_speech(padded, 16000)

            speech_frames = 0
            total_frames = 0
            for i in range(0, len(audio_chunk) - frame_size + 1, frame_size):
                total_frames += 1
                if self._vad.is_speech(audio_chunk[i : i + frame_size], 16000):
                    speech_frames += 1
            return (
                speech_frames / max(total_frames, 1)
            ) >= settings.webrtc_speech_ratio_threshold

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _infer)
