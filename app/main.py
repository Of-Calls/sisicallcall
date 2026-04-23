import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# 서브모듈 import 이전에 .env 로드 — Google/Twilio 등 외부 서비스 자격증명이
# os.environ 경유로 읽히므로 반드시 최상단에서 수행해야 한다.
load_dotenv()

from fastapi import FastAPI

from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.core.middleware import RequestLoggingMiddleware
from app.api.v1 import auth, call, summary, tenant, dashboard
from app.utils.config import settings
from app.utils.logger import get_logger

_logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # XTTS 사용 시 모델을 미리 로드해 첫 통화의 greeting 지연(약 30초) 방지.
    # cache_node 의 BGE-M3 는 module-level 인스턴스화로 이미 자동 warm-up 됨.
    # XTTS warm-up + greeting 사전 합성 캐시
    if (settings.tts_provider or "google").lower() == "xtts":
        from app.services.tts.channel import tts_channel
        try:
            tts = tts_channel._get_tts()
            if hasattr(tts, "_ensure_model"):
                _logger.info("XTTS 모델 warm-up 시작 (서버 첫 통화 지연 방지)")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, tts._ensure_model)
                _logger.info("XTTS 모델 warm-up 완료 — 통화 응답 준비됨")
        except Exception as e:
            _logger.error("XTTS warm-up 실패 — 첫 통화 시 lazy load 됨: %s", e)

        # Greeting 사전 합성 — 통화 연결 즉시 greeting 재생을 위해 μ-law 캐시
        if hasattr(tts_channel._get_tts(), "warm_cache"):
            try:
                import asyncpg, json as _json
                _DEFAULT_GREETING = "안녕하세요, 고객센터입니다. 무엇을 도와드릴까요?"
                _DEFAULT_OFFHOURS = (
                    "안녕하세요, 고객센터입니다. "
                    "현재 상담원 운영 시간이 아니지만 기본적인 문의는 도와드릴 수 있습니다. "
                    "무엇을 도와드릴까요?"
                )
                greeting_texts: set[str] = {_DEFAULT_GREETING, _DEFAULT_OFFHOURS}

                # 테넌트별 커스텀 greeting 수집
                conn = await asyncpg.connect(settings.database_url)
                try:
                    rows = await conn.fetch(
                        "SELECT settings FROM tenants WHERE settings IS NOT NULL"
                    )
                    for row in rows:
                        data = row["settings"]
                        if isinstance(data, str):
                            data = _json.loads(data)
                        for field in ("greeting", "offhours_greeting"):
                            text = (data or {}).get(field, "").strip()
                            if text:
                                greeting_texts.add(text)
                finally:
                    await conn.close()

                _tts = tts_channel._get_tts()
                for text in greeting_texts:
                    _logger.info("Greeting 사전 합성 중 text_len=%d", len(text))
                    await _tts.warm_cache(text)
                _logger.info("Greeting 캐시 완료 — %d개 텍스트 사전 합성됨", len(greeting_texts))
            except Exception as e:
                _logger.error("Greeting 캐시 실패 — 첫 통화 시 실시간 합성: %s", e)

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
