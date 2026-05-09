import asyncio

import azure.cognitiveservices.speech as speechsdk

from app.services.tts.base import BaseTTSService
from app.utils.config import settings


class AzureTTSService(BaseTTSService):

    def __init__(self):
        self._speech_config = speechsdk.SpeechConfig(
            subscription=settings.azure_speech_key,
            region=settings.azure_speech_region,
        )
        self._speech_config.speech_synthesis_voice_name = settings.azure_tts_voice
        # Twilio Media Stream 호환: mulaw 8kHz 8-bit
        self._speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw
        )

    async def synthesize(self, text: str) -> bytes:
        if not text:
            return b""

        # 결정적 후처리 — '~' 가 LLM 응답에 남으면 Azure TTS 가 "에서" 로 어색하게 읽음.
        # prompt 룰로도 차단하지만 LLM 비결정성 안전망. 'X~Y' → 'X에서 Y'.
        text = text.replace("~", "에서 ")

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=None,  # 스피커 출력 비활성화 → 바이트만 반환
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: synthesizer.speak_text_async(text).get()
        )

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise RuntimeError(f"Azure TTS 합성 실패: {result.reason}")

        return result.audio_data
