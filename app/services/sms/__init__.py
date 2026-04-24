from app.services.sms.base import BaseSMSService


def get_sms_service() -> BaseSMSService:
    from app.utils.config import settings
    provider = (settings.sms_provider or "solapi").lower()
    if provider == "twilio":
        from app.services.sms.twilio import TwilioSMSService
        return TwilioSMSService()
    from app.services.sms.solapi import SolapiSMSService
    return SolapiSMSService()
