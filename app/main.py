import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

# torch / pydantic settings 이전: .env 만 읽어 스레드·OpenMP 상한 선적용 (compare_runtime_bootstrap 는 settings 미사용)
from app.services.speaker_verify.compare_runtime_bootstrap import (
    bootstrap_speaker_compare_runtime_env_before_imports,
)

bootstrap_speaker_compare_runtime_env_before_imports(log=True)

from app.utils.config import settings
from app.services.speaker_verify.compare_runtime_env import apply_speaker_compare_runtime_env

if settings.speaker_verify_compare_enabled:
    apply_speaker_compare_runtime_env(log=True)

from fastapi import FastAPI

from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.core.middleware import RequestLoggingMiddleware
from app.api.v1 import auth, call, post_call, summary, tenant, dashboard, voiceprint
from app.api.v1.oauth import router as oauth_router
from app.utils.logger import get_logger

_logger = get_logger(__name__)

# NeMo baseline 로드만 전용(기본 asyncio executor 와 분리 → 기동 시 다른 run_in_executor 와 경합 완화)
_NEMO_COMPARE_BASELINE_EXECUTOR: ThreadPoolExecutor | None = None


def _nemo_compare_baseline_executor() -> ThreadPoolExecutor:
    global _NEMO_COMPARE_BASELINE_EXECUTOR
    if _NEMO_COMPARE_BASELINE_EXECUTOR is None:
        _NEMO_COMPARE_BASELINE_EXECUTOR = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nemo-compare-baseline",
        )
    return _NEMO_COMPARE_BASELINE_EXECUTOR


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


def _preload_nemo_mel_only_sync() -> None:
    """NeMo mel 선로드(설정 시). ONNX 는 백그라운드."""
    _safe_preload_nemo_mel_sync()


def _preload_finetuned_and_parallel_onnx_sync() -> None:
    """기동 시 파인튜닝 ONNX 로드. NeMo mel 워밍업은 설정 시에만 병렬 제출."""
    from app.services.speaker_verify.onnx_pipeline import get_finetuned_onnx_service

    if settings.warmup_nemo_mel_at_startup:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_nemo = pool.submit(_safe_preload_nemo_mel_sync)
            f_ft = pool.submit(get_finetuned_onnx_service)
            f_nemo.result()
            f_ft.result()
    else:
        get_finetuned_onnx_service()


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
            "startup: 파인튜닝 ONNX(+선택 NeMo mel) 동시 로딩… "
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
            "startup: NeMo mel(설정 시)만 즉시 — 파인튜닝 ONNX는 백그라운드 "
            "(기동 시 ONNX까지 즉시 올리려면 PRELOAD_FINETUNED_NEMO_AT_STARTUP=true)"
        )
        await loop.run_in_executor(None, _preload_nemo_mel_only_sync)
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

    if settings.speaker_verify_compare_enabled:
        from app.services.speaker_verify.titanet_compare import (
            get_titanet_compare_speaker_verify_service,
        )

        bl_onnx = (settings.speaker_verify_compare_baseline_onnx_path or "").strip()
        bl_src = (
            bl_onnx
            if bl_onnx
            else "models/speech_verification/titanet-s.onnx (기본)"
        )
        _logger.info(
            "startup: SPEAKER_VERIFY_COMPARE_ENABLED — baseline ONNX 워커에서 백그라운드 로드 (%s). "
            "완료 전에는 compare·CSV·baseline 게이트 비활성( finetuned ONNX 만 )",
            bl_src,
        )

        async def _load_compare_baseline_background() -> None:
            try:
                svc = get_titanet_compare_speaker_verify_service()
                _logger.info(
                    "startup: baseline ONNX 백그라운드 로드 — 전용 스레드 풀(1)에 제출"
                )
                await loop.run_in_executor(
                    _nemo_compare_baseline_executor(),
                    svc.load_baseline_model,
                )
                if svc.load_error:
                    _logger.error(
                        "startup: titanet_compare baseline ONNX 실패 — 비교·게이트는 finetuned 경로로 폴백: %s",
                        svc.load_error,
                    )
                else:
                    _logger.info(
                        "startup: titanet_compare baseline ONNX 백그라운드 로드 완료 "
                        "(순정 ONNX + finetuned ONNX 비교·게이트 사용 가능)"
                    )
            except Exception:
                _logger.exception("startup: titanet_compare baseline ONNX 백그라운드 로드 예외")

        asyncio.create_task(_load_compare_baseline_background())

    if settings.call_debug_routes_enabled:
        from app.services.speaker_verify.titanet_compare import _resolve_compare_csv_path

        _logger.info(
            "startup: CALL_DEBUG_ROUTES_ENABLED — 통화 후 결과: 브라우저 "
            "`/call/debug/verify-compare?format=html` | 웹훅 안내 `/call/debug/twilio-webhook-hint` | "
            "CSV=%s",
            _resolve_compare_csv_path(),
        )

    if settings.reset_voiceprint_on_startup:
        from app.services.speaker_verify import enrollment as voice_enrollment_reset
        from app.services.speaker_verify.onnx_pipeline import clear_all_onnx_voiceprints
        from app.services.speaker_verify.titanet_compare import clear_all_compare_voiceprints

        clear_all_onnx_voiceprints()
        clear_all_compare_voiceprints()
        voice_enrollment_reset.reset_all_enrollment_state()
        _logger.info(
            "startup: RESET_VOICEPRINT_ON_STARTUP=true — ONNX voiceprint·enrollment 전역 비움"
        )

    yield

    global _NEMO_COMPARE_BASELINE_EXECUTOR
    ex = _NEMO_COMPARE_BASELINE_EXECUTOR
    if ex is not None:
        try:
            ex.shutdown(wait=True)
        except Exception:
            _logger.exception("startup: compare baseline ONNX 전용 executor shutdown 실패")
        _NEMO_COMPARE_BASELINE_EXECUTOR = None


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
