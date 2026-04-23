from abc import ABC, abstractmethod


class BaseAuthService(ABC):
    @abstractmethod
    async def verify_face(
        self, image_bytes: bytes, tenant_id: str, customer_ref: str
    ) -> tuple[bool, float]:
        """ArcFace 얼굴 인증.

        Returns:
            (verified, similarity_score)
            score = -1.0 if face not detected, -2.0 if no stored embedding
        """
        raise NotImplementedError

    @abstractmethod
    async def register_face(
        self, image_bytes: bytes, tenant_id: str, customer_ref: str
    ) -> None:
        """얼굴 임베딩 추출 후 face_embeddings 저장 (upsert)."""
        raise NotImplementedError
