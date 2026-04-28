from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.api.v1.oauth import router as oauth_router

app = FastAPI(title="Sisicallcall OAuth Only")
app.include_router(oauth_router, prefix="/api/v1/oauth", tags=["oauth"])