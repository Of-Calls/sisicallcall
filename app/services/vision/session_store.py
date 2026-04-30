from datetime import datetime, timedelta, timezone

import redis.asyncio as aioredis

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

VISION_STATUS_WAITING_UPLOAD = "WAITING_UPLOAD"


def _upload_key(token_hash: str) -> str:
    return f"vision_upload:{token_hash}"


def _call_session_key(call_id: str) -> str:
    return f"call:{call_id}:session"


class VisionUploadSessionStore:
    def __init__(self) -> None:
        self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    async def create_upload_session(
        self,
        token_hash: str,
        call_id: str,
        tenant_id: str,
        caller_number: str,
        expires_in_sec: int,
    ) -> dict:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=expires_in_sec)
        data = {
            "call_id": call_id,
            "tenant_id": tenant_id,
            "caller_number": caller_number,
            "status": VISION_STATUS_WAITING_UPLOAD,
            "used": "false",
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
        }
        key = _upload_key(token_hash)
        await self._redis.hset(key, mapping=data)
        await self._redis.expire(key, expires_in_sec)
        logger.info(
            "vision upload session created call_id=%s tenant_id=%s ttl=%d",
            call_id,
            tenant_id,
            expires_in_sec,
        )
        return data

    async def get_upload_session(self, token_hash: str) -> dict | None:
        key = _upload_key(token_hash)
        data = await self._redis.hgetall(key)
        if not data:
            return None
        return data

    async def set_call_vision_status(self, call_id: str, status: str) -> None:
        key = _call_session_key(call_id)
        try:
            exists = await self._redis.exists(key)
            if not exists:
                logger.info(
                    "call session does not exist; skip vision_status save call_id=%s status=%s",
                    call_id,
                    status,
                )
                return
            await self._redis.hset(key, "vision_status", status)
        except Exception as e:
            logger.warning(
                "call vision status save failed call_id=%s status=%s: %s",
                call_id,
                status,
                e,
            )
