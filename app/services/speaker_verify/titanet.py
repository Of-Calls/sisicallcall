"""TitaNet 화자 검증 서비스 — 대영(R-01) 연구 결과로 채택.

백엔드:
  - TITANET_ONNX_PATH 가 파일로 존재하면 ONNX(mel 입력 [B,80,T]) + pcm→mel 전처리.
  - 없으면 NeMo EncDecSpeakerLabelModel.from_pretrained (PCM 직접 입력).

동작:
    첫 발화 누적(settings.titanet_enrollment_sec 초) → voiceprint 등록
    이후 발화 → 코사인 유사도 비교(≥ settings.titanet_similarity_threshold → 동일 화자)

voiceprint 미등록 시 bypass 모드:
    is_speaker_verified=True 강제 반환 (feature_spec.md §3.2 설계와 동일)
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time

import numpy as np
import torch

from app.services.speaker_verify.base import BaseSpeakerVerifyService
from app.services.speaker_verify.titanet_mel import (
    TitaNetMelFrontend,
    pcm16_bytes_to_float_mono,
)
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SAMPLE_RATE = 16000  # STT 노드와 동일한 16kHz PCM 입력

# NeMo 모델 캐시 위치 — ONNX 미사용 시에만 의미 있음
os.environ.setdefault("NEMO_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".nemo_cache"))

_MIN_MEL_FRAMES = 8


def _onnx_embedding_from_output(out: np.ndarray) -> np.ndarray:
    """ONNX 출력 텐서를 1D 임베딩으로 통일."""
    a = np.asarray(out, dtype=np.float32)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        return a[0]
    if a.ndim == 3:
        # [B, D, T] 가 일반적 — 시간 평균
        if a.shape[1] <= a.shape[2]:
            return a[0].mean(axis=-1)
        return a[0].mean(axis=1)
    return a.reshape(-1)


class TitaNetSpeakerVerifyService(BaseSpeakerVerifyService):
    """TitaNet 기반 화자 검증 서비스.

    per-call voiceprint 인메모리 관리.
    통화 종료 시 반드시 cleanup(call_id) 호출로 메모리 해제.
    """

    def __init__(self) -> None:
        self._voiceprints: dict[str, np.ndarray] = {}
        onnx_path = (settings.titanet_onnx_path or "").strip()
        if onnx_path and Path(onnx_path).is_file():
            self._backend = "onnx"
            self._init_onnx(onnx_path)
        else:
            if onnx_path:
                logger.warning(
                    "TITANET_ONNX_PATH 설정됐지만 파일 없음 → NeMo 경로 사용 path=%s",
                    onnx_path,
                )
            self._backend = "nemo"
            self._init_nemo()

    def _init_onnx(self, onnx_path: str) -> None:
        import onnxruntime as ort

        t0 = time.monotonic()
        logger.info("TitaNet ONNX 로딩 시작 path=%s …", onnx_path)
        providers: list[str] = []
        if torch.cuda.is_available():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self._ort_sess = ort.InferenceSession(onnx_path, providers=providers)
        ins = self._ort_sess.get_inputs()
        if not ins:
            raise RuntimeError("ONNX 모델에 입력이 없습니다.")
        self._onnx_in_name = ins[0].name
        # mel 은 CPU 에서 계산 (경량, VRAM·ORT CUDA 와 분리)
        self._mel = TitaNetMelFrontend(torch.device("cpu"))
        elapsed = time.monotonic() - t0
        logger.info(
            "TitaNet ONNX 준비 완료 elapsed=%.2fs providers=%s in=%s",
            elapsed,
            self._ort_sess.get_providers(),
            self._onnx_in_name,
        )

    def _init_nemo(self) -> None:
        from tqdm import tqdm
        from nemo.collections.asr.models import EncDecSpeakerLabelModel  # noqa: E402

        logger.info(
            "TitaNet NeMo 로드 중 model=%s device=%s",
            settings.titanet_model_name,
            _DEVICE,
        )
        t0 = time.monotonic()
        with tqdm(
            total=3,
            desc="TitaNet 로드",
            unit="step",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]",
            dynamic_ncols=True,
        ) as pbar:
            tqdm.write(
                "TitaNet: from_pretrained… (1/3, CPU면 수 분 걸릴 수 있음 — NeMo 캐시시 이후는 짧아짐)"
            )
            self._model = EncDecSpeakerLabelModel.from_pretrained(
                model_name=settings.titanet_model_name
            )
            pbar.update(1)
            pbar.set_postfix_str("checkpoint->device")
            self._model = self._model.to(_DEVICE)
            pbar.update(1)
            pbar.set_postfix_str("eval")
            self._model.eval()
            pbar.update(1)
        self._ort_sess = None
        self._mel = None
        elapsed = time.monotonic() - t0
        logger.info(
            "TitaNet NeMo 로드 완료 model=%s elapsed=%.1fs",
            settings.titanet_model_name,
            elapsed,
        )

    def _mel_pad_if_needed(self, mel: np.ndarray) -> np.ndarray:
        if mel.ndim != 3 or mel.shape[1] != 80:
            return mel
        tdim = mel.shape[2]
        if tdim >= _MIN_MEL_FRAMES:
            return mel
        pad = _MIN_MEL_FRAMES - tdim
        return np.pad(mel, ((0, 0), (0, 0), (0, pad)), mode="constant")

    def _extract_embedding_sync(self, audio_chunk: bytes) -> np.ndarray:
        """동기 임베딩 추출 — run_in_executor 로 호출."""
        if self._backend == "onnx":
            assert self._ort_sess is not None and self._mel is not None
            samples = pcm16_bytes_to_float_mono(audio_chunk)
            mel = self._mel(samples)
            mel = self._mel_pad_if_needed(mel)
            mel = np.ascontiguousarray(mel, dtype=np.float32)
            feeds = {self._onnx_in_name: mel}
            outs = self._ort_sess.run(None, feeds)
            if not outs:
                raise RuntimeError("ONNX 출력 없음")
            raw = outs[0]
            emb = _onnx_embedding_from_output(raw)
            return emb

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
                call_id,
                embedding.shape[0],
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
                call_id,
                similarity,
                settings.titanet_similarity_threshold,
                is_verified,
            )
            return is_verified, similarity
        except Exception as e:
            logger.error("화자 검증 실패 call_id=%s: %s", call_id, e)
            return False, 0.0

    def cleanup(self, call_id: str) -> None:
        """통화 종료 시 voiceprint 메모리 해제."""
        if self._voiceprints.pop(call_id, None) is not None:
            logger.info("voiceprint 삭제 call_id=%s", call_id)


_singleton: TitaNetSpeakerVerifyService | None = None


def get_titanet_service() -> TitaNetSpeakerVerifyService:
    """enrollment_node · speaker_verify_node 가 공유하는 모듈 레벨 싱글톤."""
    global _singleton
    if _singleton is None:
        _singleton = TitaNetSpeakerVerifyService()
    return _singleton
