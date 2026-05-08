from __future__ import annotations

from dataclasses import dataclass

from app.services.auth.session import AuthSessionService
from app.services.sms import get_sms_service
from app.services.sms.base import BaseSMSService
from app.utils.auth_sms import build_ocr_auth_sms, ocr_auth_url
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class OCRAuthLinkResult:
    auth_id: str
    status: str
    message: str
    link: str
    sent: bool


class OCRAuthLinkService:
    def __init__(
        self,
        *,
        session_service: AuthSessionService | None = None,
        sms_service: BaseSMSService | None = None,
    ) -> None:
        self._session_svc = session_service or AuthSessionService()
        self._sms_svc = sms_service or get_sms_service()

    async def create_session_and_send_link(
        self,
        *,
        tenant_id: str,
        customer_ref: str,
        customer_phone: str,
        call_id: str,
    ) -> OCRAuthLinkResult:
        auth_id = await self._session_svc.create_session(
            tenant_id=tenant_id,
            customer_ref=customer_ref,
            customer_phone=customer_phone,
            call_id=call_id,
        )
        sent = await self.send_link(auth_id, customer_phone)
        message = (
            "SMS 스킵 모드 — OCR 인증 링크: " + ocr_auth_url(auth_id)
            if settings.auth_skip_sms
            else ("OCR 인증 SMS 발송 완료" if sent else "OCR 인증 SMS 발송 실패 — 인증 세션은 유효")
        )
        return OCRAuthLinkResult(
            auth_id=auth_id,
            status="pending",
            message=message,
            link=ocr_auth_url(auth_id),
            sent=sent,
        )

    async def send_link(self, auth_id: str, customer_phone: str) -> bool:
        if settings.auth_skip_sms:
            logger.info("OCR 인증 SMS 스킵 auth_id=%s link=%s", auth_id, ocr_auth_url(auth_id))
            return True

        sent = await self._sms_svc.send_sms(
            to=customer_phone,
            body=build_ocr_auth_sms(auth_id),
        )
        if not sent:
            logger.error(
                "OCR 인증 SMS 발송 실패 auth_id=%s phone=%s",
                auth_id,
                customer_phone,
            )
        return sent
