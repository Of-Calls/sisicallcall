from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.core.middleware import RequestLoggingMiddleware
from app.api.v1 import auth, call, post_call, summary, tenant, dashboard
from app.api.v1.oauth import router as oauth_router
from app.utils.logger import get_logger

_logger = get_logger(__name__)


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(call.router,      prefix="/call",       tags=["call"])
app.include_router(post_call.router, prefix="/post-call",  tags=["post-call"])
app.include_router(summary.router,   prefix="/summary",    tags=["summary"])
app.include_router(tenant.router,    prefix="/tenant",     tags=["tenant"])
app.include_router(dashboard.router, prefix="/dashboard",  tags=["dashboard"])
app.include_router(auth.router,      prefix="/auth",       tags=["auth"])
app.include_router(oauth_router,     prefix="/api/v1/oauth", tags=["oauth"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": APP_TITLE}
