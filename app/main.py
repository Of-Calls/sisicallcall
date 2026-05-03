import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.core.middleware import RequestLoggingMiddleware
from app.api.v1 import auth, call, post_call, summary, tenant, dashboard, voiceprint
from app.api.v1.oauth import router as oauth_router
from app.utils.config import settings
from app.utils.logger import get_logger

_logger = get_logger(__name__)


def _preload_nemo_mel_if_configured_sync() -> None:
    """TITANET_MEL_BACKEND=nemo 이고 WARMUP_NEMO_MEL_AT_STARTUP=true 일 때만 NeMo restore."""
    import torch

    from app.services.speaker_verify.titanet_mel_nemo import nemo_mel_backend_enabled, get_shared_nemo_preprocessor

    if not settings.warmup_nemo_mel_at_startup:
        return
    if nemo_mel_backend_enabled():
        get_shared_nemo_preprocessor(torch.device("cpu"))


def _safe_preload_nemo_mel_sync() -> None:
    try:
        _preload_nemo_mel_if_configured_sync()
    except Exception:
        _logger.exception(
            "NeMo mel preprocessor 선로드 실패 — torchaudio mel로 계속합니다. "
            "(TITANET_MEL_BACKEND / TITANET_SPEAKER_NEMO_PATH 확인)"
        )


def _preload_medium_and_maybe_background_finetuned_sync() -> None:
    """medium ONNX 즉시 로드 (파인튜닝은 별도 백그라운드 태스크)."""
    from app.services.speaker_verify.onnx_pipeline import get_onnx_pipeline_service

    _safe_preload_nemo_mel_sync()
    get_onnx_pipeline_service()


def _preload_finetuned_and_parallel_onnx_sync() -> None:
    """기동 시 파인튜닝 ONNX + medium ONNX 로드. NeMo mel 워밍업은 설정 시에만 병렬 제출."""
    from app.services.speaker_verify.onnx_pipeline import (
        get_finetuned_onnx_service,
        get_onnx_pipeline_service,
    )

    if settings.warmup_nemo_mel_at_startup:
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_nemo = pool.submit(_safe_preload_nemo_mel_sync)
            f_medium = pool.submit(get_onnx_pipeline_service)
            f_ft = pool.submit(get_finetuned_onnx_service)
            f_nemo.result()
            f_medium.result()
            f_ft.result()
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_medium = pool.submit(get_onnx_pipeline_service)
            f_ft = pool.submit(get_finetuned_onnx_service)
            f_medium.result()
            f_ft.result()


async def _background_warm_finetuned_onnx() -> None:
    from app.services.speaker_verify.onnx_pipeline import get_finetuned_onnx_service

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, get_finetuned_onnx_service)
        _logger.info("백그라운드 파인튜닝 ONNX 로드 완료")
    except Exception:
        _logger.exception("백그라운드 파인튜닝 ONNX 로드 실패")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    stop_hb = asyncio.Event()

    async def _heartbeat() -> None:
        n = 0
        while True:
            try:
                await asyncio.wait_for(stop_hb.wait(), timeout=45)
                return
            except asyncio.TimeoutError:
                n += 1
                _logger.warning(
                    "모델 로딩 중… (%d회째, 경과≈%ds)",
                    n,
                    n * 45,
                )

    if settings.preload_finetuned_nemo_at_startup:
        _logger.info(
            "startup: medium + 파인튜닝 ONNX 동시 로딩… "
            "(끄려면 PRELOAD_FINETUNED_NEMO_AT_STARTUP=false)"
        )
        hb_task = asyncio.create_task(_heartbeat())
        try:
            await loop.run_in_executor(None, _preload_finetuned_and_parallel_onnx_sync)
        finally:
            stop_hb.set()
        await hb_task
        _logger.info("startup: ONNX 통화 파이프라인 준비 완료")
    else:
        _logger.info(
            "startup: medium ONNX 즉시 로딩 — 파인튜닝 ONNX는 백그라운드 "
            "(기동 시 둘 다 올리려면 PRELOAD_FINETUNED_NEMO_AT_STARTUP=true)"
        )
        await loop.run_in_executor(None, _preload_medium_and_maybe_background_finetuned_sync)
        asyncio.create_task(_background_warm_finetuned_onnx())
        _logger.info("startup: 서버 수신 가능 — 파인튜닝 ONNX는 백그라운드에서 로드됨")

    from app.services.speaker_verify.titanet_mel_nemo import nemo_mel_backend_enabled

    if nemo_mel_backend_enabled() and not settings.warmup_nemo_mel_at_startup:
        _logger.info(
            "startup: TITANET_MEL_BACKEND=nemo — NeMo 는 첫 mel 사용 시 restore "
            "(기동 단축). 기동 시 미리 올리려면 WARMUP_NEMO_MEL_AT_STARTUP=true"
        )

    from app.services.speaker_verify.onnx_pipeline import log_onnx_inference_session_check

    log_onnx_inference_session_check()

    if settings.reset_voiceprint_on_startup:
        from app.services.speaker_verify import enrollment as voice_enrollment_reset
        from app.services.speaker_verify.onnx_pipeline import clear_all_onnx_voiceprints

        clear_all_onnx_voiceprints()
        voice_enrollment_reset.reset_all_enrollment_state()
        _logger.info(
            "startup: RESET_VOICEPRINT_ON_STARTUP=true — ONNX voiceprint·enrollment 전역 비움"
        )

    yield


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(call.router, prefix="/call", tags=["call"])
app.include_router(post_call.router, prefix="/post-call", tags=["post-call"])
app.include_router(summary.router, prefix="/summary", tags=["summary"])
app.include_router(tenant.router, prefix="/tenant", tags=["tenant"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(oauth_router, prefix="/api/v1/oauth", tags=["oauth"])
app.include_router(voiceprint.router, prefix="/api/v1/voiceprint", tags=["voiceprint"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": APP_TITLE}
