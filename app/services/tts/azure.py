"""Azure Speech SDK TTS provider — 한국어 Neural Voice, μ-law 8kHz 네이티브 출력.

Twilio Media Stream 가 요구하는 μ-law 8kHz mono 포맷을 Azure가 직접 출력하므로
audioop 변환(ratecv → lin2ulaw) 단계가 사라지고, 합성 시간 자체도 200~400ms 수준.

reference:
    https://learn.microsoft.com/azure/ai-services/speech-service/get-started-text-to-speech
    SpeechSynthesisOutputFormat 목록 — Raw8Khz8BitMonoMULaw 가 Twilio 호환 포맷.

voice 옵션 (.env AZURE_TTS_VOICE 로 변경):
    ko-KR-SunHiNeural   여성·따뜻 (고객센터 적합, 기본값)
    ko-KR-InJoonNeural  남성·차분
    ko-KR-YuJinNeural   여성·활발
    ko-KR-HyunsuNeural  남성·청년
"""
import asyncio

import azure.cognitiveservices.speech as speechsdk

from app.services.tts.base import BaseTTSService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_OUTPUT_FORMAT = speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw


class AzureTTSService(BaseTTSService):
    def __init__(self):
        self._config: speechsdk.SpeechConfig | None = None

    def _ensure_config(self) -> speechsdk.SpeechConfig:
        if self._config is not None:
            return self._config
        if not settings.azure_speech_key or not settings.azure_speech_region:
            raise RuntimeError(
                "Azure Speech 자격증명 누락 — .env 의 AZURE_SPEECH_KEY, "
                "AZURE_SPEECH_REGION 설정 필수"
            )
        cfg = speechsdk.SpeechConfig(
            subscription=settings.azure_speech_key,
            region=settings.azure_speech_region,
        )
        cfg.speech_synthesis_voice_name = settings.azure_tts_voice
        cfg.set_speech_synthesis_output_format(_OUTPUT_FORMAT)
        self._config = cfg
        logger.info(
            "Azure TTS 준비 완료 voice=%s region=%s format=Raw8Khz8BitMonoMULaw",
            settings.azure_tts_voice, settings.azure_speech_region,
        )
        return self._config

    def _synthesize_sync(self, text: str) -> bytes:
        cfg = self._ensure_config()
        # audio_config=None → 스피커 출력 비활성, result.audio_data 로 메모리에 수신.
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)
        result = synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return bytes(result.audio_data)
        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            raise RuntimeError(
                f"Azure TTS 합성 취소 reason={details.reason} "
                f"error={details.error_details}"
            )
        raise RuntimeError(f"Azure TTS 합성 실패 result.reason={result.reason}")

    async def synthesize(self, text: str) -> bytes:
        """텍스트 → μ-law 8kHz mono bytes (Twilio Media Stream 호환).

        Azure Speech SDK 가 직접 Raw8Khz8BitMonoMULaw 로 출력 → 변환 단계 없음.
        """
        if not text:
            return b""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)
