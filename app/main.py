from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.core.middleware import RequestLoggingMiddleware
from app.api.v1 import auth, call, post_call, summary, tenant, dashboard, vision, ocr
from app.api.v1.oauth import router as oauth_router
from app.services.embedding import get_embedder
from app.utils.config import settings
from app.utils.logger import get_logger

_logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _logger.info("startup: loading embedding model (provider=%s)...", settings.embedding_provider)
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
app.include_router(vision.router, prefix="/vision", tags=["vision"])
app.include_router(ocr.router, tags=["ocr"])
app.include_router(oauth_router, prefix="/api/v1/oauth", tags=["oauth"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": APP_TITLE}
