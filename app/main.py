import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# 서브모듈 import 이전에 .env 로드 — Google/Twilio 등 외부 서비스 자격증명이
# os.environ 경유로 읽히므로 반드시 최상단에서 수행해야 한다.
load_dotenv()

from fastapi import FastAPI

from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.core.middleware import RequestLoggingMiddleware
from app.api.v1 import call, summary, tenant, dashboard
from app.utils.config import settings
from app.utils.logger import get_logger

_logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # XTTS 사용 시 모델을 미리 로드해 첫 통화의 greeting 지연(약 30초) 방지.
    # cache_node 의 BGE-M3 는 module-level 인스턴스화로 이미 자동 warm-up 됨.
    if (settings.tts_provider or "google").lower() == "xtts":
        from app.services.tts.channel import tts_channel
        try:
            tts = tts_channel._get_tts()  # XTTSService 인스턴스 생성
            if hasattr(tts, "_ensure_model"):
                _logger.info("XTTS 모델 warm-up 시작 (서버 첫 통화 지연 방지)")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, tts._ensure_model)
                _logger.info("XTTS 모델 warm-up 완료 — 통화 응답 준비됨")
        except Exception as e:
            _logger.error("XTTS warm-up 실패 — 첫 통화 시 lazy load 됨: %s", e)
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


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": APP_TITLE}
