import asyncio

from app.services.vad.base import BaseVADService

WEBRTC_MODE = (
    3  # 0(관대함. 노이즈도 speech로 감지) ~ 3(엄격함. 조금 애매하면 무음처리함)
)

# 한 프레임 길이 (30ms)
WEBRTC_FRAME_MS = 30
# 발화가 끝났다고 판단할 최소 발화 비율 (30% 이상)
WEBRTC_SPEECH_RATIO_THRESHOLD = 0.3


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

            self._vad = webrtcvad.Vad(WEBRTC_MODE)
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
                return False
            frame_size = int(16000 * WEBRTC_FRAME_MS / 1000) * 2
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
            ) >= WEBRTC_SPEECH_RATIO_THRESHOLD

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _infer)
