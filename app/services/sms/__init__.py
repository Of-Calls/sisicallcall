from app.services.sms.base import BaseSMSService


def get_sms_service() -> BaseSMSService:
    from app.utils.config import settings
    provider = (settings.sms_provider or "mock").strip().lower()
    if provider == "mock":
        from app.services.sms.mock import MockSMSService
        return MockSMSService()
    if provider == "twilio":
        from app.services.sms.twilio import TwilioSMSService
        return TwilioSMSService()
    if provider == "solapi":
        from app.services.sms.solapi import SolapiSMSService
        return SolapiSMSService()
    raise ValueError(f"Unsupported SMS_PROVIDER: {provider}")
