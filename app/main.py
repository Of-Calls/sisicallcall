from dotenv import load_dotenv

# 서브모듈 import 이전에 .env 로드 — Google/Twilio 등 외부 서비스 자격증명이
# os.environ 경유로 읽히므로 반드시 최상단에서 수행해야 한다.
load_dotenv()

from fastapi import FastAPI

from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.core.middleware import RequestLoggingMiddleware
from app.api.v1 import call, summary, tenant, dashboard

app = FastAPI(title=APP_TITLE, version=APP_VERSION, description=APP_DESCRIPTION)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(call.router, prefix="/call", tags=["call"])
app.include_router(summary.router, prefix="/summary", tags=["summary"])
app.include_router(tenant.router, prefix="/tenant", tags=["tenant"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": APP_TITLE}
