from abc import ABC, abstractmethod


class BaseSMSService(ABC):
    @abstractmethod
    async def send_sms(self, to: str, body: str) -> bool:
        """SMS 발송. to: E.164 형식 (+821012345678)"""
        raise NotImplementedError
