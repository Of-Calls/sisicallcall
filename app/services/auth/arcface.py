import asyncio
import io

import asyncpg
import numpy as np
from PIL import Image, ImageOps

from app.services.auth.base import BaseAuthService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class ArcFaceAuthService(BaseAuthService):
    def __init__(self) -> None:
        self._face_app = None

    def _get_face_app(self):
        if self._face_app is None:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name=settings.arcface_model_name)
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._face_app = app
            logger.info("ArcFace 모델 로드 완료 model=%s", settings.arcface_model_name)
        return self._face_app

    def _extract_embedding_sync(self, image_bytes: bytes) -> np.ndarray | None:
        img = np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB"))
        faces = self._get_face_app().get(img)
        if not faces:
            return None
        return faces[0].embedding.astype(np.float32)

    async def verify_face(
        self, image_bytes: bytes, tenant_id: str, customer_ref: str
    ) -> tuple[bool, float]:
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, self._extract_embedding_sync, image_bytes)
        if embedding is None:
            logger.warning("얼굴 감지 실패 tenant=%s ref=%s", tenant_id, customer_ref)
            return False, -1.0

        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                "SELECT embedding FROM face_embeddings WHERE tenant_id=$1 AND customer_ref=$2",
                tenant_id,
                customer_ref,
            )
        finally:
            await conn.close()

        if row is None:
            logger.warning("등록된 임베딩 없음 tenant=%s ref=%s", tenant_id, customer_ref)
            return False, -2.0

        stored = np.array(row["embedding"], dtype=np.float32)
        sim = _cosine_similarity(embedding, stored)
        threshold = settings.arcface_similarity_threshold
        verified = sim >= threshold
        logger.info(
            "얼굴 인증 sim=%.4f threshold=%.2f verified=%s tenant=%s ref=%s",
            sim, threshold, verified, tenant_id, customer_ref,
        )
        return verified, sim

    async def register_face(
        self, image_bytes: bytes, tenant_id: str, customer_ref: str
    ) -> None:
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, self._extract_embedding_sync, image_bytes)
        if embedding is None:
            raise ValueError("얼굴 감지 실패 — 등록 불가")

        conn = await asyncpg.connect(settings.database_url)
        try:
            await conn.execute(
                """
                INSERT INTO face_embeddings (tenant_id, customer_ref, embedding)
                VALUES ($1, $2, $3)
                ON CONFLICT (tenant_id, customer_ref)
                DO UPDATE SET embedding = $3, updated_at = NOW()
                """,
                tenant_id,
                customer_ref,
                embedding.tolist(),
            )
        finally:
            await conn.close()
        logger.info("얼굴 임베딩 등록 완료 tenant=%s ref=%s", tenant_id, customer_ref)
