import re

from app.services.sms.base import BaseSMSService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_UPLOAD_URL_RE = re.compile(r"((?:https?://|/)[^\s]+/v/)[^\s]+")


def _redact_upload_url(value: str) -> str:
    return _UPLOAD_URL_RE.sub(r"\1<redacted>", value)


class MockSMSService(BaseSMSService):
    async def send_sms(self, to: str, body: str) -> bool:
        logger.info("mock SMS send to=%s body=%s", to, _redact_upload_url(body))
        return True
