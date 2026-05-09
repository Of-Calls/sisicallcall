from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.core.middleware import RequestLoggingMiddleware
from app.api.v1 import admin_auth, auth, call, call_history, dashboard, post_call, summary, tenant, vision
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
                        _logger.warning("ChromaDB warmup failed tenant=%s: %s", tid[:8], e)
                _logger.info("startup: ChromaDB warm (%d/%d tenants)", warmed, len(rows))
            except Exception as e:
                _logger.warning("ChromaDB warmup failed: %s", e)
    else:
        _logger.info("startup: warmup disabled (WARMUP_ENABLED=false)")

    yield


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

app.include_router(call.router,      prefix="/call",       tags=["call"])
app.include_router(call_history.router, prefix="/call",    tags=["call-history"])
app.include_router(post_call.router, prefix="/post-call",  tags=["post-call"])
app.include_router(summary.router,   prefix="/summary",    tags=["summary"])
app.include_router(tenant.router,    prefix="/tenant",     tags=["tenant"])
app.include_router(dashboard.router, prefix="/dashboard",  tags=["dashboard"])
app.include_router(admin_auth.router, prefix="/auth",      tags=["admin-auth"])
app.include_router(auth.router,      prefix="/auth",       tags=["auth"])
app.include_router(vision.router,    prefix="/vision",     tags=["vision"])
app.include_router(oauth_router,     prefix="/api/v1/oauth", tags=["oauth"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": APP_TITLE}
