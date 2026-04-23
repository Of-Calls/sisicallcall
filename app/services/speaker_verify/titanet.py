"""TitaNet 화자 검증 서비스 — 대영(R-01) 연구 결과로 채택.

동작:
    첫 발화 누적(settings.titanet_enrollment_sec 초) → voiceprint 등록
    이후 발화 → 코사인 유사도 비교(≥ settings.titanet_similarity_threshold → 동일 화자)

voiceprint 미등록 시 bypass 모드:
    is_speaker_verified=True 강제 반환 (feature_spec.md §3.2 설계와 동일)

GPU 자동 감지:
    CUDA 가용 시 GPU, 아니면 CPU (팀원 환경별 자동 폴백)
"""
import asyncio
import os

import numpy as np
import torch

# NeMo 모델 캐시 위치 — 절대 경로 대신 프로젝트 상대 경로 사용
os.environ.setdefault("NEMO_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".nemo_cache"))

from nemo.collections.asr.models import EncDecSpeakerLabelModel  # noqa: E402

from app.services.speaker_verify.base import BaseSpeakerVerifyService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SAMPLE_RATE = 16000  # STT 노드와 동일한 16kHz PCM 입력


class TitaNetSpeakerVerifyService(BaseSpeakerVerifyService):
    """TitaNet 기반 화자 검증 서비스.

    per-call voiceprint 인메모리 관리.
    통화 종료 시 반드시 cleanup(call_id) 호출로 메모리 해제.
    """

    def __init__(self) -> None:
        self._voiceprints: dict[str, np.ndarray] = {}
        logger.info(
            "TitaNet 모델 로드 중 model=%s device=%s",
            settings.titanet_model_name, _DEVICE,
        )
        self._model = EncDecSpeakerLabelModel.from_pretrained(
            model_name=settings.titanet_model_name
        )
        self._model = self._model.to(_DEVICE)
        self._model.eval()
        logger.info("TitaNet 모델 로드 완료 model=%s", settings.titanet_model_name)

    def _extract_embedding_sync(self, audio_chunk: bytes) -> np.ndarray:
        """동기 임베딩 추출 — run_in_executor 로 호출."""
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_signal = torch.tensor(samples).unsqueeze(0).to(_DEVICE)
        audio_len = torch.tensor([audio_signal.shape[1]]).to(_DEVICE)
        with torch.no_grad():
            _, emb = self._model.forward(
                input_signal=audio_signal, input_signal_length=audio_len
            )
        return emb.squeeze().cpu().numpy()

    async def _extract_embedding(self, audio_chunk: bytes) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._extract_embedding_sync, audio_chunk)

    async def extract_and_store(self, audio_chunk: bytes, call_id: str) -> None:
        """첫 발화 누적 오디오로 voiceprint 등록."""
        try:
            embedding = await self._extract_embedding(audio_chunk)
            self._voiceprints[call_id] = embedding
            logger.info(
                "voiceprint 등록 완료 call_id=%s dim=%d",
                call_id, embedding.shape[0],
            )
        except Exception as e:
            logger.error("voiceprint 추출 실패 call_id=%s: %s", call_id, e)

    async def verify(self, audio_chunk: bytes, call_id: str) -> tuple[bool, float]:
        """화자 검증. voiceprint 미등록 시 bypass(True, 1.0) 반환."""
        if call_id not in self._voiceprints:
            logger.debug("voiceprint 없음 → bypass call_id=%s", call_id)
            return True, 1.0

        try:
            embedding = await self._extract_embedding(audio_chunk)
            voiceprint = self._voiceprints[call_id]
            similarity = float(
                np.dot(embedding, voiceprint)
                / (np.linalg.norm(embedding) * np.linalg.norm(voiceprint))
            )
            is_verified = similarity >= settings.titanet_similarity_threshold
            logger.info(
                "화자 검증 call_id=%s similarity=%.4f threshold=%.2f verified=%s",
                call_id, similarity, settings.titanet_similarity_threshold, is_verified,
            )
            return is_verified, similarity
        except Exception as e:
            logger.error("화자 검증 실패 call_id=%s: %s", call_id, e)
            return False, 0.0

    def cleanup(self, call_id: str) -> None:
        """통화 종료 시 voiceprint 메모리 해제."""
        if self._voiceprints.pop(call_id, None) is not None:
            logger.info("voiceprint 삭제 call_id=%s", call_id)
