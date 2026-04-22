import asyncio

from app.services.embedding.base import BaseEmbeddingService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BGEM3LocalEmbeddingService(BaseEmbeddingService):
    """BGE-M3 로컬 추론 구현체 — 팀장(희원) 테스트용.
    희영 연구 결과에 따라 BGEM3APIEmbeddingService 또는 이 클래스 중 선택.
    """

    def __init__(self):
        from FlagEmbedding import BGEM3FlagModel
        logger.info("BGE-M3 로컬 모델 로딩 중 (BAAI/bge-m3)...")
        self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        logger.info("BGE-M3 로컬 모델 로딩 완료")

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        output = self._model.encode(texts, batch_size=12, max_length=512)
        return output["dense_vecs"].tolist()

    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self._encode_sync, [text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode_sync, texts)
