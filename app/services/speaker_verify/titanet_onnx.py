"""
TitaNet ONNX 기반 화자 검증 서비스 — NeMo 의존성 없음.

사전 준비:
    scripts/export_titanet.py 로 models/titanet_small.onnx 또는
    models/titanet_large.onnx 를 생성해 두어야 합니다.

mel spectrogram 전처리 파라미터는 export 시 함께 저장된
*_meta.json 을 자동으로 읽습니다. meta 파일이 없으면 NeMo 기본값을 사용합니다.
"""

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.services.speaker_verify.base import BaseSpeakerVerifyService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SAMPLE_RATE = 16000

# NeMo TitaNet 기본 전처리 파라미터
_DEFAULT_META = {
    "sample_rate":    16000,
    "n_fft":          512,
    "n_mels":         80,
    "win_length":     400,   # 25 ms @ 16 kHz
    "hop_length":     160,   # 10 ms @ 16 kHz
    "window":         "hann",
    "dither":         0.0,
    "log_zero_guard": 2 ** -24,
}


def _load_meta(onnx_path: str) -> dict:
    meta_path = onnx_path.replace(".onnx", "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    logger.warning(f"meta 파일 없음 ({meta_path}), NeMo 기본값 사용")
    return _DEFAULT_META


def _preprocess(audio_bytes: bytes, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """PCM int16 bytes → (mel_features, length) — NeMo AudioToMelSpectrogramPreprocessor 동일."""
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # dither
    if meta["dither"] > 0.0:
        samples += meta["dither"] * np.random.randn(*samples.shape).astype(np.float32)

    # STFT
    import librosa  # 런타임 import — onnxruntime 전용 환경에서 librosa만 추가하면 됨
    stft = librosa.stft(
        samples,
        n_fft=meta["n_fft"],
        hop_length=meta["hop_length"],
        win_length=meta["win_length"],
        window=meta["window"],
        center=True,
    )
    power = np.abs(stft) ** 2

    # mel filterbank
    mel_fb = librosa.filters.mel(
        sr=meta["sample_rate"],
        n_fft=meta["n_fft"],
        n_mels=meta["n_mels"],
    )
    mel = mel_fb @ power

    # log
    guard = float(meta["log_zero_guard"])
    mel = np.log(mel + guard).astype(np.float32)

    # per-feature normalization (NeMo normalize="per_feature")
    mean = mel.mean(axis=1, keepdims=True)
    std = mel.std(axis=1, keepdims=True) + 1e-5
    mel = (mel - mean) / std

    # (n_mels, T) → (1, n_mels, T)  +  length tensor
    mel = mel[np.newaxis, :, :]                        # (1, 80, T)
    length = np.array([mel.shape[2]], dtype=np.int64)  # (1,)
    return mel, length


class TitaNetOnnxSpeakerVerifyService(BaseSpeakerVerifyService):
    def __init__(self, model_path: str | None = None) -> None:
        if model_path is None:
            name = settings.titanet_model_name  # "titanet_large" or "titanet_small"
            model_path = f"models/{name}.onnx"

        model_path = str(Path(model_path))
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX 모델 파일이 없습니다: {model_path}\n"
                "  → scripts/export_titanet.py 를 NeMo 환경에서 먼저 실행하세요."
            )

        self._meta = _load_meta(model_path)
        self._voiceprints: dict[str, np.ndarray] = {}

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_names = [i.name for i in self._session.get_inputs()]

        logger.info(f"TitaNet ONNX 로드 완료: {model_path} (providers={providers})")

    def _extract_embedding_sync(self, audio_bytes: bytes) -> np.ndarray:
        mel, length = _preprocess(audio_bytes, self._meta)

        # NeMo export 시 입력: (processed_signal, processed_signal_length)
        feeds = {
            self._input_names[0]: mel,
            self._input_names[1]: length,
        }
        outputs = self._session.run(None, feeds)
        # NeMo forward: logits, emb — emb이 두 번째 출력
        emb = outputs[1].squeeze()
        return emb.astype(np.float32)

    async def _extract_embedding(self, audio_bytes: bytes) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_embedding_sync, audio_bytes)

    async def extract_and_store(self, audio_chunk: bytes, call_id: str) -> None:
        try:
            embedding = await self._extract_embedding(audio_chunk)
            self._voiceprints[call_id] = embedding
            logger.info(f"call_id={call_id} voiceprint 등록 완료 dim={embedding.shape[0]}")
        except Exception as e:
            logger.error(f"call_id={call_id} voiceprint 추출 실패: {e}")

    async def verify(self, audio_chunk: bytes, call_id: str) -> tuple[bool, float]:
        if call_id not in self._voiceprints:
            logger.warning(f"call_id={call_id} voiceprint 없음 → bypass (True)")
            return True, 1.0

        try:
            embedding = await self._extract_embedding(audio_chunk)
            voiceprint = self._voiceprints[call_id]

            norm_e = np.linalg.norm(embedding)
            norm_v = np.linalg.norm(voiceprint)
            similarity = float(np.dot(embedding, voiceprint) / (norm_e * norm_v))
            is_verified = similarity >= settings.titanet_similarity_threshold

            logger.info(
                f"call_id={call_id} similarity={similarity:.4f} "
                f"threshold={settings.titanet_similarity_threshold} verified={is_verified}"
            )
            return is_verified, similarity

        except Exception as e:
            logger.error(f"call_id={call_id} 화자 검증 실패: {e}")
            return False, 0.0

    def cleanup(self, call_id: str) -> None:
        if self._voiceprints.pop(call_id, None) is not None:
            logger.info(f"call_id={call_id} voiceprint 삭제")
