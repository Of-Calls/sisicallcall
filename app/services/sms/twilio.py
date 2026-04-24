import asyncio

from app.services.sms.base import BaseSMSService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TwilioSMSService(BaseSMSService):
    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            from twilio.rest import Client
            self._client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        return self._client

    async def send_sms(self, to: str, body: str) -> bool:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._send_sync, to, body)
            logger.info("SMS 발송 완료 to=%s", to)
            return True
        except Exception as e:
            logger.error("SMS 발송 실패 to=%s: %s", to, e)
            return False

    def _send_sync(self, to: str, body: str) -> None:
        client = self._get_client()
        client.messages.create(
            body=body,
            from_=settings.twilio_phone_number,
            to=to,
        )
