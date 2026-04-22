import asyncio
import wave
from io import BytesIO

from app.services.stt.base import BaseSTTService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DeepgramSTTService(BaseSTTService):
    def __init__(self):
        from deepgram import DeepgramClient, PrerecordedOptions

        self._client = DeepgramClient(settings.deepgram_api_key.strip())
        self._options = PrerecordedOptions(
            model="nova-2", punctuate=True, language="ko"
        )

    async def transcribe(self, audio_chunk: bytes) -> str:
        if not audio_chunk:
            return ""

        def _pcm16_to_wav_bytes(pcm16_audio: bytes, sample_rate: int = 16000) -> bytes:
            buf = BytesIO()
            with wave.open(buf, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm16_audio)
            return buf.getvalue()

        def _run() -> str:
            wav_bytes = _pcm16_to_wav_bytes(audio_chunk)
            payload = {"buffer": wav_bytes, "mimetype": "audio/wav"}
            response = self._client.listen.prerecorded.v("1").transcribe_file(
                payload, self._options
            )
            if not response or not response.results or not response.results.channels:
                return ""
            alternatives = response.results.channels[0].alternatives
            if not alternatives:
                return ""
            return (alternatives[0].transcript or "").strip()

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, _run)
        except Exception as e:
            logger.warning("Deepgram STT 호출 실패: %s", e)
            return ""
