import asyncio

from app.services.sms.base import BaseSMSService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SolapiSMSService(BaseSMSService):
    def __init__(self) -> None:
        self._service = None

    def _get_service(self):
        if self._service is None:
            from solapi import SolapiMessageService
            self._service = SolapiMessageService(
                api_key=settings.solapi_api_key,
                api_secret=settings.solapi_api_secret,
            )
        return self._service

    async def send_sms(self, to: str, body: str) -> bool:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._send_sync, to, body)
            logger.info("SMS 발송 완료 to=%s", to)
            return True
        except Exception as e:
            failed_messages = getattr(e, "failed_messages", None)
            logger.error(
                "SMS 발송 실패 to=%s err=%s failed_messages=%s",
                to,
                e,
                failed_messages,
            )
            return False

    @staticmethod
    def _to_local(number: str) -> str:
        # E.164 → 국내 형식 (+821047722480 → 01047722480)
        n = number.lstrip("+")
        if n.startswith("82"):
            n = "0" + n[2:]
        return n

    def _send_sync(self, to: str, body: str) -> None:
        from solapi.model.request.message import Message
        svc = self._get_service()
        msg = Message(
            to=self._to_local(to),
            from_=self._to_local(settings.solapi_sender_number),
            text=body,
        )
        svc.send(msg)
