import asyncio
import audioop
import os
import time
from dataclasses import dataclass

from app.services.vad.base import BaseVADService


@dataclass
class VADResult:
    model: str
    is_speech: bool
    latency_ms: float
    available: bool
    note: str = ""


def _energy_fallback(pcm16_16k: bytes, threshold: int = 220) -> bool:
    """모델이 없을 때 최소한의 음성 존재 여부를 추정한다."""
    if not pcm16_16k:
        return False
    return audioop.rms(pcm16_16k, 2) >= threshold


def _pcm16_to_float32_list(audio_chunk: bytes) -> list[float]:
    if not audio_chunk:
        return []
    samples_count = len(audio_chunk) // 2
    if samples_count <= 0:
        return []
    int_samples = [
        int.from_bytes(audio_chunk[i * 2 : i * 2 + 2], byteorder="little", signed=True)
        for i in range(samples_count)
    ]
    return [s / 32768.0 for s in int_samples]


class _BenchmarkVADService(BaseVADService):
    name = "base"

    def __init__(self) -> None:
        self.available = True
        self.note = ""

    async def detect(self, audio_chunk: bytes) -> bool:
        raise NotImplementedError


class WebRTCVADService(_BenchmarkVADService):
    name = "webrtc_vad"

    async def detect(self, audio_chunk: bytes) -> bool:
        def _infer() -> bool:
            try:
                import webrtcvad
            except Exception:
                self.available = False
                self.note = "webrtcvad 미설치, energy fallback 사용"
                return _energy_fallback(audio_chunk)

            vad = webrtcvad.Vad(2)
            frame_ms = 30
            frame_size = int(16000 * frame_ms / 1000) * 2
            if len(audio_chunk) < frame_size:
                padded = audio_chunk + b"\x00" * (frame_size - len(audio_chunk))
                return vad.is_speech(padded, 16000)

            speech_frames = 0
            total_frames = 0
            for i in range(0, len(audio_chunk) - frame_size + 1, frame_size):
                frame = audio_chunk[i : i + frame_size]
                total_frames += 1
                if vad.is_speech(frame, 16000):
                    speech_frames += 1
            return (speech_frames / max(total_frames, 1)) >= 0.3

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _infer)


class SileroV4VADService(_BenchmarkVADService):
    name = "silero_v4"

    async def detect(self, audio_chunk: bytes) -> bool:
        def _infer() -> bool:
            try:
                import torch
                from silero_vad import get_speech_timestamps
            except Exception:
                self.available = False
                self.note = "silero v4 미설치(torch 없음), energy fallback 사용"
                return _energy_fallback(audio_chunk)

            try:
                # v4 태그 우선 시도
                model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad:v4.0",
                    model="silero_vad",
                    force_reload=False,
                    onnx=False,
                    trust_repo=True,
                )
                waveform = torch.tensor(_pcm16_to_float32_list(audio_chunk), dtype=torch.float32)
                if waveform.numel() == 0:
                    return False
                speech_ts = get_speech_timestamps(waveform, model, sampling_rate=16000)
                self.available = True
                self.note = ""
                return len(speech_ts) > 0
            except Exception:
                self.available = False
                self.note = "silero v4 태그 로드 실패, energy fallback 사용"
                return _energy_fallback(audio_chunk)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _infer)


class SileroV5VADService(_BenchmarkVADService):
    name = "silero_v5"

    async def detect(self, audio_chunk: bytes) -> bool:
        def _infer() -> bool:
            try:
                import torch
                from silero_vad import get_speech_timestamps
            except Exception:
                self.available = False
                self.note = "silero v5 미설치(torch 없음), energy fallback 사용"
                return _energy_fallback(audio_chunk)

            try:
                # v5 태그 우선 시도
                model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad:v5.0",
                    model="silero_vad",
                    force_reload=False,
                    onnx=False,
                    trust_repo=True,
                )
                waveform = torch.tensor(_pcm16_to_float32_list(audio_chunk), dtype=torch.float32)
                if waveform.numel() == 0:
                    return False
                speech_ts = get_speech_timestamps(waveform, model, sampling_rate=16000)
                self.available = True
                self.note = ""
                return len(speech_ts) > 0
            except Exception:
                self.available = False
                self.note = "silero v5 태그 로드 실패, energy fallback 사용"
                return _energy_fallback(audio_chunk)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _infer)


class DeepgramVADService(_BenchmarkVADService):
    name = "deepgram_vad"

    async def detect(self, audio_chunk: bytes) -> bool:
        def _infer() -> bool:
            try:
                from deepgram import DeepgramClient, PrerecordedOptions
            except Exception:
                self.available = False
                self.note = "deepgram-sdk 미설치, energy fallback 사용"
                return _energy_fallback(audio_chunk)

            api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
            if not api_key:
                self.available = False
                self.note = "DEEPGRAM_API_KEY 없음, energy fallback 사용"
                return _energy_fallback(audio_chunk)

            try:
                client = DeepgramClient(api_key)
                payload = {"buffer": audio_chunk, "mimetype": "audio/l16;rate=16000"}
                options = PrerecordedOptions(model="nova-2", vad_events=True, punctuate=False)
                response = client.listen.prerecorded.v("1").transcribe_file(payload, options)
                transcript = (
                    response.results.channels[0].alternatives[0].transcript
                    if response and response.results and response.results.channels
                    else ""
                )
                self.available = True
                self.note = ""
                return bool(transcript and transcript.strip())
            except Exception:
                self.available = False
                self.note = "deepgram 호출 실패, energy fallback 사용"
                return _energy_fallback(audio_chunk)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _infer)


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


async def compare_vad_models(pcm16_16k: bytes) -> list[VADResult]:
    """4개 VAD 후보를 동일 오디오로 실행해 결과를 반환한다."""
    services: list[_BenchmarkVADService] = [
        WebRTCVADService(),
        SileroV4VADService(),
        SileroV5VADService(),
        DeepgramVADService(),
    ]
    jobs = [
        _timed_run(service, pcm16_16k) for service in services
    ]
    results = await asyncio.gather(*jobs)
    return sorted(results, key=lambda r: r.latency_ms)
