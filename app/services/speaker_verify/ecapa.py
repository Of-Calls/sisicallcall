import asyncio

import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

from app.services.speaker_verify.base import BaseSpeakerVerifyService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SAMPLE_RATE = 16000
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ECAPASpeakerVerifyService(BaseSpeakerVerifyService):
    def __init__(self) -> None:
        logger.info(f"ECAPA-TDNN 모델 로드 중... (source={settings.ecapa_model_source}, device={_DEVICE})")
        try:
            self._model = EncoderClassifier.from_hparams(
                source=settings.ecapa_model_source,
                savedir=settings.ecapa_model_savedir,
                run_opts={"device": _DEVICE},
            )
            self._model.eval()
            logger.info("ECAPA-TDNN 모델 로드 완료")
        except Exception as e:
            logger.error(f"ECAPA-TDNN 모델 로드 실패: {e}")
            raise
        self._voiceprints: dict[str, np.ndarray] = {}

    def _bytes_to_tensor(self, audio_chunk: bytes) -> torch.Tensor:
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.tensor(samples).unsqueeze(0)

    def _extract_embedding_sync(self, audio_chunk: bytes) -> np.ndarray:
        waveform = self._bytes_to_tensor(audio_chunk)
        with torch.no_grad():
            embedding = self._model.encode_batch(waveform)
        return embedding.squeeze().numpy()

    async def _extract_embedding(self, audio_chunk: bytes) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_embedding_sync, audio_chunk)

    async def extract_and_store(self, audio_chunk: bytes, call_id: str) -> None:
        try:
            embedding = await self._extract_embedding(audio_chunk)
            self._voiceprints[call_id] = embedding
            logger.info(
                f"call_id={call_id} voiceprint 등록 완료 "
                f"embedding_dim={embedding.shape[0]}"
            )
        except Exception as e:
            logger.error(f"call_id={call_id} voiceprint 추출 실패: {e}")

    async def verify(self, audio_chunk: bytes, call_id: str) -> tuple[bool, float]:
        if call_id not in self._voiceprints:
            logger.warning(f"call_id={call_id} voiceprint 없음 → bypass (True 반환)")
            return True, 1.0

        try:
            embedding = await self._extract_embedding(audio_chunk)
            voiceprint = self._voiceprints[call_id]

            similarity = float(
                np.dot(embedding, voiceprint)
                / (np.linalg.norm(embedding) * np.linalg.norm(voiceprint))
            )
            is_verified = similarity >= settings.ecapa_similarity_threshold

            logger.info(
                f"call_id={call_id} ECAPA similarity={similarity:.4f} "
                f"threshold={settings.ecapa_similarity_threshold} "
                f"verified={is_verified}"
            )
            return is_verified, similarity

        except Exception as e:
            logger.error(f"call_id={call_id} 화자 검증 실패: {e}")
            return False, 0.0

    def cleanup(self, call_id: str) -> None:
        if self._voiceprints.pop(call_id, None) is not None:
            logger.info(f"call_id={call_id} voiceprint 삭제 완료")
