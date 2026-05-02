import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.core.middleware import RequestLoggingMiddleware
from app.api.v1 import auth, call, post_call, summary, tenant, dashboard
from app.api.v1.oauth import router as oauth_router
from app.utils.config import settings
from app.utils.logger import get_logger

_logger = get_logger(__name__)


def _preload_titanet_sync() -> None:
    """NeMo/TitaNet 가중치 로드 — 통화 처리 구간에서는 호출하지 않음."""
    from app.services.speaker_verify.titanet import get_titanet_service

    get_titanet_service()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 화자 검증 사용 시에만 기동 시 로드 — 요청 처리 중 첫 로드로 인한 지연·WS 끊김 방지
    if settings.speaker_verify_enabled:
        _logger.info("startup: TitaNet 로딩 시작…")
        loop = asyncio.get_running_loop()
        stop_hb = asyncio.Event()

        async def _heartbeat() -> None:
            # NeMo가 GIL을 길게 잡으면 스레드 하트비트는 거의 안 돈다 → asyncio로 주기 로그
            n = 0
            while True:
                try:
                    await asyncio.wait_for(stop_hb.wait(), timeout=45)
                    return
                except asyncio.TimeoutError:
                    n += 1
                    _logger.warning(
                        "TitaNet 로딩 중… (%d회째 안내, 경과≈%ds) CPU에서는 매우 오래 걸릴 수 있습니다. "
                        "당장 서버만 필요하면 프로세스 종료 후 .env 에 SPEAKER_VERIFY_ENABLED=false",
                        n,
                        n * 45,
                    )

        hb_task = asyncio.create_task(_heartbeat())
        try:
            await loop.run_in_executor(None, _preload_titanet_sync)
        finally:
            stop_hb.set()
        await hb_task

        _logger.info("startup: TitaNet 준비 완료")
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


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": APP_TITLE}
