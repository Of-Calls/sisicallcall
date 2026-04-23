import asyncio
import time
from dataclasses import dataclass

from app.services.vad.webrtc_vad import WebRTCVADService
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VADResult:
    model: str
    is_speech: bool
    latency_ms: float
    available: bool
    note: str = ""


_WEBRTC_VAD_SERVICE = WebRTCVADService()
_VAD_SERVICES = [_WEBRTC_VAD_SERVICE]


async def preload_vad_models() -> None:
    await asyncio.gather(*(service.initialize() for service in _VAD_SERVICES))
    logger.info(
        msg=f"VAD 모델 프리로드 완료: {_WEBRTC_VAD_SERVICE.name}, 사용 가능 여부: {_WEBRTC_VAD_SERVICE.available}, 비고: {_WEBRTC_VAD_SERVICE.note}",
    )


async def _timed_run(service: WebRTCVADService, audio: bytes) -> VADResult:
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
