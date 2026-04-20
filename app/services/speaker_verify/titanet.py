import asyncio
import os

import numpy as np
import torch

os.environ.setdefault("NEMO_CACHE_DIR", "C:/torch_cache/nemo")

from nemo.collections.asr.models import EncDecSpeakerLabelModel  # noqa: E402

from app.services.speaker_verify.base import BaseSpeakerVerifyService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SAMPLE_RATE = 16000
_voiceprints: dict[str, np.ndarray] = {}


class TitaNetSpeakerVerifyService(BaseSpeakerVerifyService):
    def __init__(self) -> None:
        logger.info(f"TitaNet 모델 로드 중... (model={settings.titanet_model_name})")
        try:
            self._model = EncDecSpeakerLabelModel.from_pretrained(
                model_name=settings.titanet_model_name
            )
            self._model = self._model.to("cpu")
            self._model.eval()
            logger.info("TitaNet 모델 로드 완료")
        except Exception as e:
            logger.error(f"TitaNet 모델 로드 실패: {e}")
            raise

    def _extract_embedding_sync(self, audio_chunk: bytes) -> np.ndarray:
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_signal = torch.tensor(samples).unsqueeze(0)  # [1, T]
        audio_signal_len = torch.tensor([audio_signal.shape[1]])  # [1]
        with torch.no_grad():
            _, emb = self._model.forward(
                input_signal=audio_signal, input_signal_length=audio_signal_len
            )
        return emb.squeeze().numpy()

    async def _extract_embedding(self, audio_chunk: bytes) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_embedding_sync, audio_chunk)

    async def extract_and_store(self, audio_chunk: bytes, call_id: str) -> None:
        try:
            embedding = await self._extract_embedding(audio_chunk)
            _voiceprints[call_id] = embedding
            logger.info(
                f"call_id={call_id} voiceprint 등록 완료 "
                f"embedding_dim={embedding.shape[0]}"
            )
        except Exception as e:
            logger.error(f"call_id={call_id} voiceprint 추출 실패: {e}")

    async def verify(self, audio_chunk: bytes, call_id: str) -> tuple[bool, float]:
        if call_id not in _voiceprints:
            logger.warning(f"call_id={call_id} voiceprint 없음 → bypass (True 반환)")
            return True, 1.0

        try:
            embedding = await self._extract_embedding(audio_chunk)
            voiceprint = _voiceprints[call_id]

            similarity = float(
                np.dot(embedding, voiceprint)
                / (np.linalg.norm(embedding) * np.linalg.norm(voiceprint))
            )
            is_verified = similarity >= settings.titanet_similarity_threshold

            logger.info(
                f"call_id={call_id} TitaNet similarity={similarity:.4f} "
                f"threshold={settings.titanet_similarity_threshold} "
                f"verified={is_verified}"
            )
            return is_verified, similarity

        except Exception as e:
            logger.error(f"call_id={call_id} 화자 검증 실패: {e}")
            return False, 0.0

    def cleanup(self, call_id: str) -> None:
        if _voiceprints.pop(call_id, None) is not None:
            logger.info(f"call_id={call_id} voiceprint 삭제 완료")
