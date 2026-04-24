import hashlib
import hmac
import random

import redis.asyncio as aioredis

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_INSTRUCTIONS = ["look_left", "look_right", "look_up", "look_down", "blink"]
_LIVENESS_TOKEN_TTL = 300  # 5분


def _key(auth_id: str) -> str:
    return f"auth:liveness:{auth_id}"


class LivenessService:
    def __init__(self) -> None:
        self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    async def generate_instructions(self, auth_id: str) -> dict:
        count = min(settings.liveness_instruction_count, len(_INSTRUCTIONS))
        instructions = random.sample(_INSTRUCTIONS, count)
        payload = f"{auth_id}:{':'.join(instructions)}"
        token = hmac.new(
            settings.liveness_hmac_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        await self._redis.set(_key(auth_id), token, ex=_LIVENESS_TOKEN_TTL)
        logger.info("liveness instructions 생성 auth_id=%s count=%d", auth_id, count)
        return {"instructions": instructions, "token": token}

    async def validate_token(self, auth_id: str, token: str) -> bool:
        stored = await self._redis.get(_key(auth_id))
        if not stored:
            logger.warning("liveness token 없음 또는 만료 auth_id=%s", auth_id)
            return False
        valid = hmac.compare_digest(stored, token)
        if valid:
            await self._redis.delete(_key(auth_id))
        return valid
