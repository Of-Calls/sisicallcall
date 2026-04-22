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
_MODEL_IDS = {
    "base": "damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k",
    "v2": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
}
_voiceprints: dict[str, dict[str, np.ndarray]] = {"base": {}, "v2": {}}


class ERes2NetSpeakerVerifyService(BaseSpeakerVerifyService):
    def __init__(self, variant: str = "base") -> None:
        assert variant in ("base", "v2"), f"variant must be 'base' or 'v2', got {variant}"
        self._variant = variant
        logger.info(f"ERes2Net-{variant} 모델 로드 중...")
        try:
            from modelscope.pipelines import pipeline as ms_pipeline
            from modelscope.utils.constant import Tasks
            self._pipeline = ms_pipeline(
                task=Tasks.speaker_verification,
                model=_MODEL_IDS[variant],
            )
            logger.info(f"ERes2Net-{variant} 모델 로드 완료")
        except Exception as e:
            logger.error(f"ERes2Net-{variant} 모델 로드 실패: {e}")
            raise

    @property
    def _threshold(self) -> float:
        if self._variant == "base":
            return settings.eres2net_base_similarity_threshold
        return settings.eres2net_v2_similarity_threshold

    def _extract_embedding_sync(self, audio_chunk: bytes) -> np.ndarray:
        import torch
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, samples, _SAMPLE_RATE)
            tmp_path = f.name
        try:
            processed = self._pipeline.preprocess([tmp_path])
            with torch.no_grad():
                emb = self._pipeline.model(processed[0].unsqueeze(0))
            return emb.squeeze(0).cpu().numpy()
        finally:
            os.unlink(tmp_path)

    async def _extract_embedding(self, audio_chunk: bytes) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_embedding_sync, audio_chunk)

    async def extract_and_store(self, audio_chunk: bytes, call_id: str) -> None:
        try:
            embedding = await self._extract_embedding(audio_chunk)
            _voiceprints[self._variant][call_id] = embedding
            logger.info(
                f"call_id={call_id} ERes2Net-{self._variant} voiceprint 등록 완료 "
                f"dim={embedding.shape[0]}"
            )
        except Exception as e:
            logger.error(f"call_id={call_id} ERes2Net-{self._variant} voiceprint 추출 실패: {e}")
            raise

    async def verify(self, audio_chunk: bytes, call_id: str) -> tuple[bool, float]:
        if call_id not in _voiceprints[self._variant]:
            logger.warning(f"call_id={call_id} ERes2Net-{self._variant} voiceprint 없음 → bypass")
            return True, 1.0
        try:
            embedding = await self._extract_embedding(audio_chunk)
            voiceprint = _voiceprints[self._variant][call_id]
            similarity = float(
                np.dot(embedding, voiceprint)
                / (np.linalg.norm(embedding) * np.linalg.norm(voiceprint))
            )
            is_verified = similarity >= self._threshold
            logger.info(
                f"call_id={call_id} ERes2Net-{self._variant} similarity={similarity:.4f} "
                f"threshold={self._threshold} verified={is_verified}"
            )
            return is_verified, similarity
        except Exception as e:
            logger.error(f"call_id={call_id} ERes2Net-{self._variant} 화자 검증 실패: {e}")
            return False, 0.0

    def cleanup(self, call_id: str) -> None:
        if _voiceprints[self._variant].pop(call_id, None) is not None:
            logger.info(f"call_id={call_id} ERes2Net-{self._variant} voiceprint 삭제 완료")
