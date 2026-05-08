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
from app.services.speaker_verify.compare_runtime_env import (
    apply_speaker_compare_runtime_env,
)

if settings.speaker_verify_compare_enabled:
    apply_speaker_compare_runtime_env(log=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.core.middleware import RequestLoggingMiddleware
from app.api.v1 import (
    admin_auth,
    auth,
    call,
    call_history,
    dashboard,
    post_call,
    summary,
    tenant,
    vision,
)
from app.api.v1.oauth import router as oauth_router
from app.services.embedding import get_embedder
from app.utils.config import settings
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

    from app.services.speaker_verify.titanet_mel_nemo import (
        nemo_mel_backend_enabled,
        get_shared_nemo_preprocessor,
    )

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
    _logger.info(
        "startup: loading embedding model (provider=%s)...", settings.embedding_provider
    )
    get_embedder()
    _logger.info("startup: embedding model ready")

    _logger.info("startup: warming up speaker verify (TitaNet-L ONNX)...")
    from app.services.speaker_verify import get_speaker_verify_service

    await get_speaker_verify_service().warmup()
    _logger.info("startup: speaker verify ready")

    _logger.info("startup: warming up BM25 indices for active tenants...")
    from app.services.retrieval import prewarm_all_tenants

    await prewarm_all_tenants()
    _logger.info("startup: BM25 ready")

    _logger.info("startup: prewarming TTS filler audios...")
    from app.services.tts.azure import AzureTTSService
    from app.services.tts.filler import prewarm_fillers

    await prewarm_fillers(AzureTTSService())
    _logger.info("startup: filler ready")

    # Cold start warmup — Qwen3 첫 inference / OpenAI httpx / ChromaDB per-tenant.
    # 첫 통화 첫 turn latency ~2.3s 단축. fail-tolerant — 워밍 실패해도 startup 진행.
    if settings.warmup_enabled:
        _logger.info("startup: warming up embedding model (first inference)...")
        dummy_emb: list[float] | None = None
        try:
            dummy_emb = await get_embedder().embed_query("warmup")
            _logger.info("startup: embedding model warm")
        except Exception as e:
            _logger.warning("embedding warmup failed: %s", e)

        _logger.info("startup: warming up OpenAI client...")
        try:
            from app.services.llm.gpt4o_mini import GPT4OMiniService

            await GPT4OMiniService().generate("ping", "ok", max_tokens=5)
            _logger.info("startup: OpenAI client warm")
        except Exception as e:
            _logger.warning("OpenAI warmup failed: %s", e)

        if dummy_emb:
            _logger.info("startup: warming up ChromaDB per-tenant...")
            try:
                import asyncpg
                from app.services.rag.chroma import ChromaRAGService

                rag = ChromaRAGService()
                conn = await asyncpg.connect(settings.database_url)
                try:
                    rows = await conn.fetch("SELECT id FROM tenants")
                finally:
                    await conn.close()
                warmed = 0
                for r in rows:
                    tid = str(r["id"])
                    try:
                        await rag.search_with_meta(dummy_emb, tid, top_k=1)
                        warmed += 1
                    except Exception as e:
                        _logger.warning(
                            "ChromaDB warmup failed tenant=%s: %s", tid[:8], e
                        )
                _logger.info(
                    "startup: ChromaDB warm (%d/%d tenants)", warmed, len(rows)
                )
            except Exception as e:
                _logger.warning("ChromaDB warmup failed: %s", e)
    else:
        _logger.info("startup: warmup disabled (WARMUP_ENABLED=false)")

    yield

    global _NEMO_COMPARE_BASELINE_EXECUTOR
    ex = _NEMO_COMPARE_BASELINE_EXECUTOR
    if ex is not None:
        try:
            ex.shutdown(wait=True)
        except Exception:
            _logger.exception(
                "startup: compare baseline ONNX 전용 executor shutdown 실패"
            )
        _NEMO_COMPARE_BASELINE_EXECUTOR = None


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ],
    # Keep this False while the frontend stores access tokens in localStorage.
    # If refresh-token cookies are added later, switch to True only with
    # explicit origins, secure cookie settings, and matching SameSite policy.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(call.router, prefix="/call", tags=["call"])
app.include_router(call_history.router, prefix="/call", tags=["call-history"])
app.include_router(post_call.router, prefix="/post-call", tags=["post-call"])
app.include_router(summary.router, prefix="/summary", tags=["summary"])
app.include_router(tenant.router, prefix="/tenant", tags=["tenant"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
app.include_router(admin_auth.router, prefix="/auth", tags=["admin-auth"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(vision.router, prefix="/vision", tags=["vision"])
app.include_router(oauth_router, prefix="/api/v1/oauth", tags=["oauth"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": APP_TITLE}
