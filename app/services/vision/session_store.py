from datetime import datetime, timedelta, timezone

import redis.asyncio as aioredis

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

VISION_STATUS_WAITING_UPLOAD = "WAITING_UPLOAD"
VISION_STATUS_PROCESSING = "PROCESSING"
VISION_STATUS_DONE = "DONE"
VISION_STATUS_NEED_CONFIRM = "NEED_CONFIRM"
VISION_STATUS_FAILED = "FAILED"


class VisionUploadSessionNotFoundError(Exception):
    pass


class VisionUploadSessionAlreadyUsedError(Exception):
    pass


RESERVE_UPLOAD_SESSION_LUA = """
local key = KEYS[1]
local now = ARGV[1]

if redis.call("EXISTS", key) == 0 then
  return {"missing"}
end

local used = redis.call("HGET", key, "used")
if used == "true" then
  return {"used"}
end

redis.call("HSET", key,
  "used", "true",
  "status", "PROCESSING",
  "processing_started_at", now
)

local data = redis.call("HGETALL", key)
table.insert(data, 1, "ok")
return data
"""


def _upload_key(token_hash: str) -> str:
    return f"vision_upload:{token_hash}"


def _call_session_key(call_id: str) -> str:
    return f"call:{call_id}:session"


def _decode_redis_value(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _flat_list_to_dict(items: list) -> dict:
    decoded = [_decode_redis_value(item) for item in items]
    return dict(zip(decoded[0::2], decoded[1::2]))


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

    async def reserve_upload_session_for_processing(self, token_hash: str) -> dict:
        key = _upload_key(token_hash)
        now = datetime.now(timezone.utc).isoformat()
        result = await self._redis.eval(RESERVE_UPLOAD_SESSION_LUA, 1, key, now)
        if not result:
            raise VisionUploadSessionNotFoundError()

        tag = _decode_redis_value(result[0])
        if tag == "missing":
            raise VisionUploadSessionNotFoundError()
        if tag == "used":
            raise VisionUploadSessionAlreadyUsedError()
        if tag != "ok":
            raise RuntimeError(f"unexpected reserve upload session result: {tag}")

        return _flat_list_to_dict(result[1:])

    async def attach_upload_file_info(
        self,
        token_hash: str,
        image_path: str,
        mime_type: str,
        size_bytes: int,
    ) -> None:
        key = _upload_key(token_hash)
        await self._redis.hset(
            key,
            mapping={
                "image_path": image_path,
                "mime_type": mime_type,
                "size_bytes": str(size_bytes),
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            },
        )

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
