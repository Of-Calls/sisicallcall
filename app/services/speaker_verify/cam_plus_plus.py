import asyncio
import os
import tempfile

import numpy as np
import soundfile as sf

from app.services.speaker_verify.base import BaseSpeakerVerifyService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SAMPLE_RATE = 16000
_MODEL_ID = "damo/speech_campplus_sv_zh-cn_16k-common"
_voiceprints: dict[str, np.ndarray] = {}


class CAMPlusPlusSpeakerVerifyService(BaseSpeakerVerifyService):
    def __init__(self) -> None:
        logger.info("CAM++ 모델 로드 중...")
        try:
            from modelscope.pipelines import pipeline as ms_pipeline
            from modelscope.utils.constant import Tasks
            self._pipeline = ms_pipeline(
                task=Tasks.speaker_verification,
                model=_MODEL_ID,
            )
            logger.info("CAM++ 모델 로드 완료")
        except Exception as e:
            logger.error(f"CAM++ 모델 로드 실패: {e}")
            raise

    def _extract_embedding_sync(self, audio_chunk: bytes) -> np.ndarray:
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, samples, _SAMPLE_RATE)
            tmp_path = f.name
        try:
            result = self._pipeline(tmp_path)
            return np.array(result["spk_embedding"])
        finally:
            os.unlink(tmp_path)

    async def _extract_embedding(self, audio_chunk: bytes) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_embedding_sync, audio_chunk)

    async def extract_and_store(self, audio_chunk: bytes, call_id: str) -> None:
        try:
            embedding = await self._extract_embedding(audio_chunk)
            _voiceprints[call_id] = embedding
            logger.info(f"call_id={call_id} CAM++ voiceprint 등록 완료 dim={embedding.shape[0]}")
        except Exception as e:
            logger.error(f"call_id={call_id} CAM++ voiceprint 추출 실패: {e}")
            raise

    async def verify(self, audio_chunk: bytes, call_id: str) -> tuple[bool, float]:
        if call_id not in _voiceprints:
            logger.warning(f"call_id={call_id} CAM++ voiceprint 없음 → bypass")
            return True, 1.0
        try:
            embedding = await self._extract_embedding(audio_chunk)
            voiceprint = _voiceprints[call_id]
            similarity = float(
                np.dot(embedding, voiceprint)
                / (np.linalg.norm(embedding) * np.linalg.norm(voiceprint))
            )
            is_verified = similarity >= settings.cam_similarity_threshold
            logger.info(
                f"call_id={call_id} CAM++ similarity={similarity:.4f} "
                f"threshold={settings.cam_similarity_threshold} verified={is_verified}"
            )
            return is_verified, similarity
        except Exception as e:
            logger.error(f"call_id={call_id} CAM++ 화자 검증 실패: {e}")
            return False, 0.0

    def cleanup(self, call_id: str) -> None:
        if _voiceprints.pop(call_id, None) is not None:
            logger.info(f"call_id={call_id} CAM++ voiceprint 삭제 완료")
