from abc import ABC, abstractmethod


class BaseSpeakerVerifyService(ABC):
    @abstractmethod
    async def verify(self, audio_chunk: bytes, call_id: str) -> tuple[bool, float]:
        """화자 검증 — (동일 화자 여부, cosine similarity) 반환.
        voiceprint 미등록 시 bypass: (True, 1.0) 반환.
        """
        raise NotImplementedError

    @abstractmethod
    async def extract_and_store(self, audio_chunk: bytes, call_id: str) -> None:
        """첫 발화 누적 오디오로 voiceprint 등록."""
        raise NotImplementedError

    @abstractmethod
    def cleanup(self, call_id: str) -> None:
        """통화 종료 시 voiceprint 메모리 해제."""
        raise NotImplementedError
