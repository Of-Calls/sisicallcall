from app.services.tts.base import BaseTTSService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GoogleTTSService(BaseTTSService):
    """한국어 Neural2 음성. Twilio Media Streams 호환 포맷(μ-law 8kHz mono) 직접 요청."""

    # TODO(tenant.settings 이관): 업종별 voice 다양화
    VOICE_NAME = "ko-KR-Neural2-B"
    LANGUAGE_CODE = "ko-KR"
    SPEAKING_RATE = 1.0

    def __init__(self):
        # 지연 초기화 — 자격증명은 실제 synthesize() 호출 시점에만 필요하도록.
        # 앱 import 시점에 GOOGLE_APPLICATION_CREDENTIALS 가 로드 안 돼있어도 크래시 금지.
        from google.cloud import texttospeech
        self._types = texttospeech
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = self._types.TextToSpeechAsyncClient()
        return self._client

    async def synthesize(self, text: str) -> bytes:
        """텍스트 → μ-law 8kHz mono 바이트. Twilio Media Stream outbound 에 그대로 사용."""
        client = self._ensure_client()
        response = await client.synthesize_speech(
            input=self._types.SynthesisInput(text=text),
            voice=self._types.VoiceSelectionParams(
                language_code=self.LANGUAGE_CODE,
                name=self.VOICE_NAME,
            ),
            audio_config=self._types.AudioConfig(
                audio_encoding=self._types.AudioEncoding.MULAW,
                sample_rate_hertz=8000,
                speaking_rate=self.SPEAKING_RATE,
            ),
        )
        return response.audio_content

    async def synthesize_and_stream(self, text: str) -> None:
        """(deprecated) 기존 tts_node 호환용. RFC 001 v0.2 이후는 TTSOutputChannel 경유.

        호출 시 경고 로그만 남기고 synthesize() 로 위임. 바이트는 반환하지 않음 —
        스트리밍 대상이 없으므로 no-op.
        """
        logger.warning("synthesize_and_stream is deprecated; use TTSOutputChannel")
        await self.synthesize(text)
