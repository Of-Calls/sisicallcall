import asyncio

import numpy as np

from app.services.stt.base import BaseSTTService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class WhisperSTTService(BaseSTTService):
    """faster-whisper 기반 STT 서비스."""

    def __init__(self) -> None:
        from faster_whisper import WhisperModel  # type: ignore[reportMissingImports]

        # 로컬 CPU 환경 기준 기본 설정
        self._model = WhisperModel("base", compute_type="int8")

    async def transcribe(self, audio_chunk: bytes) -> str:
        if not audio_chunk:
            return ""

        def _run() -> str:
            # PCM16(16kHz, mono) -> float32 waveform(-1.0 ~ 1.0)
            pcm16 = np.frombuffer(audio_chunk, dtype=np.int16)
            if pcm16.size == 0:
                return ""
            waveform = pcm16.astype(np.float32) / 32768.0

            segments, _ = self._model.transcribe(
                waveform,
                language="ko",
                beam_size=1,
                vad_filter=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            return text

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, _run)
        except Exception as e:
            logger.warning("Whisper STT 호출 실패: %s", e)
            return ""
