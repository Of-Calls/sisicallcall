import asyncio
import audioop
import time
import wave
from io import BytesIO
from dataclasses import dataclass

from app.services.vad.base import BaseVADService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
DEEPGRAM_SAMPLE_RATE = 16000
DEEPGRAM_CHANNELS = 1
DEEPGRAM_SAMPLE_WIDTH_BYTES = 2  # 16-bit

# VAD 민감도 튜닝 상수
ENERGY_FALLBACK_THRESHOLD = 0.5
WEBRTC_MODE = 3  # 0(덜 공격적) ~ 3(더 공격적)
WEBRTC_FRAME_MS = 30
WEBRTC_SPEECH_RATIO_THRESHOLD = 0.5
SILERO_V4_THRESHOLD = 0.5
SILERO_V5_THRESHOLD = 0.5


@dataclass
class VADResult:
    model: str
    is_speech: bool
    latency_ms: float
    available: bool
    note: str = ""


def _energy_fallback(
    pcm16_16k: bytes, threshold: int = ENERGY_FALLBACK_THRESHOLD
) -> bool:
    if not pcm16_16k:
        return False
    return audioop.rms(pcm16_16k, 2) >= threshold


def _pcm16_to_float32_list(audio_chunk: bytes) -> list[float]:
    if not audio_chunk:
        return []
    count = len(audio_chunk) // 2
    if count <= 0:
        return []
    values = [
        int.from_bytes(audio_chunk[i * 2 : i * 2 + 2], byteorder="little", signed=True)
        for i in range(count)
    ]
    return [v / 32768.0 for v in values]


def _pcm16_to_wav_bytes(
    audio_chunk: bytes, sample_rate: int = DEEPGRAM_SAMPLE_RATE
) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(DEEPGRAM_CHANNELS)
        wav_file.setsampwidth(DEEPGRAM_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_chunk)
    return buf.getvalue()


class _BenchmarkVADService(BaseVADService):
    name = "base"

    def __init__(self) -> None:
        self.available = True
        self.note = ""
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True

    async def detect(self, audio_chunk: bytes) -> bool:
        raise NotImplementedError


class WebRTCVADService(_BenchmarkVADService):
    name = "webrtc_vad"

    def __init__(self) -> None:
        super().__init__()
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
                return _energy_fallback(audio_chunk)
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


class SileroV4VADService(_BenchmarkVADService):
    name = "silero_v4"

    def __init__(self) -> None:
        super().__init__()
        self._torch = None
        self._get_speech_timestamps = None
        self._model = None

    async def initialize(self) -> None:
        if self._initialized:
            return
        try:
            import torch
            from silero_vad import get_speech_timestamps

            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad:v4.0",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            self._torch = torch
            self._get_speech_timestamps = get_speech_timestamps
            self._model = model
            self.available = True
            self.note = ""
        except Exception:
            self.available = False
            self.note = "silero v4 로드 실패, energy fallback 사용"
        finally:
            self._initialized = True

    async def detect(self, audio_chunk: bytes) -> bool:
        if not self._initialized:
            await self.initialize()

        def _infer() -> bool:
            if not (self._torch and self._get_speech_timestamps and self._model):
                return _energy_fallback(audio_chunk)
            waveform = self._torch.tensor(
                _pcm16_to_float32_list(audio_chunk), dtype=self._torch.float32
            )
            if waveform.numel() == 0:
                return False
            speech_ts = self._get_speech_timestamps(
                waveform,
                self._model,
                sampling_rate=16000,
                threshold=SILERO_V4_THRESHOLD,
            )
            return len(speech_ts) > 0

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _infer)


class SileroV5VADService(_BenchmarkVADService):
    name = "silero_v5"

    def __init__(self) -> None:
        super().__init__()
        self._torch = None
        self._get_speech_timestamps = None
        self._model = None

    async def initialize(self) -> None:
        if self._initialized:
            return
        try:
            import torch
            from silero_vad import get_speech_timestamps

            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad:v5.0",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            self._torch = torch
            self._get_speech_timestamps = get_speech_timestamps
            self._model = model
            self.available = True
            self.note = ""
        except Exception:
            self.available = False
            self.note = "silero v5 로드 실패, energy fallback 사용"
        finally:
            self._initialized = True

    async def detect(self, audio_chunk: bytes) -> bool:
        if not self._initialized:
            await self.initialize()

        def _infer() -> bool:
            if not (self._torch and self._get_speech_timestamps and self._model):
                return _energy_fallback(audio_chunk)
            waveform = self._torch.tensor(
                _pcm16_to_float32_list(audio_chunk), dtype=self._torch.float32
            )
            if waveform.numel() == 0:
                return False
            speech_ts = self._get_speech_timestamps(
                waveform,
                self._model,
                sampling_rate=16000,
                threshold=SILERO_V5_THRESHOLD,
            )
            return len(speech_ts) > 0

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _infer)


class DeepgramVADService(_BenchmarkVADService):
    name = "deepgram_vad"

    def __init__(self) -> None:
        super().__init__()
        self._client = None
        self._options = None

    async def initialize(self) -> None:
        if self._initialized:
            return
        try:
            from deepgram import DeepgramClient, PrerecordedOptions
        except Exception:
            self.available = False
            self.note = "deepgram-sdk 미설치, energy fallback 사용"
            self._initialized = True
            return

        api_key = settings.deepgram_api_key.strip()
        if not api_key:
            self.available = False
            self.note = "DEEPGRAM_API_KEY 없음, energy fallback 사용"
            self._initialized = True
            return

        try:
            self._client = DeepgramClient(api_key)
            self._options = PrerecordedOptions(model="nova-2", punctuate=False)
            self.available = True
            self.note = ""
        except Exception as e:
            self.available = False
            self.note = f"deepgram 초기화 실패: {e}"
            logger.warning("Deepgram VAD 초기화 실패: %s", e)
        finally:
            self._initialized = True

    async def detect(self, audio_chunk: bytes) -> bool:
        if not self._initialized:
            await self.initialize()

        def _infer() -> bool:
            if not (self._client and self._options):
                return _energy_fallback(audio_chunk)
            if (
                len(audio_chunk)
                < (DEEPGRAM_SAMPLE_RATE // 10) * DEEPGRAM_SAMPLE_WIDTH_BYTES
            ):
                # 너무 짧은 청크(약 100ms 미만)는 API 호출 대신 로컬 fallback 사용
                return _energy_fallback(audio_chunk)
            try:
                # Deepgram 전달 포맷: 16kHz / mono / 16-bit WAV
                wav_bytes = _pcm16_to_wav_bytes(
                    audio_chunk, sample_rate=DEEPGRAM_SAMPLE_RATE
                )
                payload = {
                    "buffer": wav_bytes,
                    "mimetype": "audio/wav",
                }
                response = self._client.listen.prerecorded.v("1").transcribe_file(
                    payload, self._options
                )
                transcript = (
                    response.results.channels[0].alternatives[0].transcript
                    if response and response.results and response.results.channels
                    else ""
                )
                return bool(transcript and transcript.strip())
            except Exception as e:
                self.note = f"deepgram 호출 실패: {e}"
                logger.warning("Deepgram VAD 호출 실패: %s", e)
                return _energy_fallback(audio_chunk)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _infer)


_VAD_SERVICES: list[_BenchmarkVADService] = [
    WebRTCVADService(),
    SileroV4VADService(),
    SileroV5VADService(),
    DeepgramVADService(),
]


async def preload_vad_models() -> None:
    await asyncio.gather(*(service.initialize() for service in _VAD_SERVICES))
    logger.info(
        "VAD 모델 프리로드 완료: %s",
        [
            {"모델": s.name, "사용가능": s.available, "비고": s.note}
            for s in _VAD_SERVICES
        ],
    )


async def _timed_run(service: _BenchmarkVADService, audio: bytes) -> VADResult:
    start = time.perf_counter()
    is_speech = await service.detect(audio)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return VADResult(
        model=service.name,
        is_speech=is_speech,
        latency_ms=round(elapsed_ms, 3),
        available=service.available,
        note=service.note,
    )


async def compare_vad_models(
    pcm16_16k: bytes,
    *,
    excluded_models: set[str] | None = None,
) -> list[VADResult]:
    excluded = excluded_models or set()
    services = [service for service in _VAD_SERVICES if service.name not in excluded]
    jobs = [_timed_run(service, pcm16_16k) for service in services]
    results = await asyncio.gather(*jobs)
    return sorted(results, key=lambda r: r.latency_ms)
