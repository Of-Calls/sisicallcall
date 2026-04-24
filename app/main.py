import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# 서브모듈 import 이전에 .env 로드 — Azure/Twilio/Deepgram 등 외부 서비스 자격증명이
# os.environ 경유로 읽히므로 반드시 최상단에서 수행해야 한다.
load_dotenv()

from fastapi import FastAPI

from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.core.middleware import RequestLoggingMiddleware
from app.api.v1 import auth, call, summary, tenant, dashboard
from app.utils.logger import get_logger

_logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Azure TTS 는 클라우드 SDK 호출 (200~400ms) — 모델 로딩/사전 합성 캐시 불필요.
    # cache_node 의 BGE-M3 는 module-level 인스턴스화로 이미 자동 warm-up 됨.

    # TitaNet warm-up — enrollment_node · speaker_verify_node 공유 싱글톤 초기화
    try:
        _logger.info("TitaNet 모델 warm-up 시작")
        from app.services.speaker_verify.titanet import get_titanet_service
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, get_titanet_service)
        _logger.info("TitaNet 모델 warm-up 완료 — 화자 검증 준비됨")
    except Exception as e:
        _logger.error("TitaNet warm-up 실패 — 첫 발화 시 lazy load 됨: %s", e)

    yield


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(call.router, prefix="/call", tags=["call"])
app.include_router(summary.router, prefix="/summary", tags=["summary"])
app.include_router(tenant.router, prefix="/tenant", tags=["tenant"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": APP_TITLE}
