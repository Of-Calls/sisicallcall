"""NeMo `EncDecSpeakerLabelModel.preprocessor` 기반 mel (ONNX 입력 [1, n_mels, T] 정렬).

`TITANET_MEL_BACKEND=nemo` 이고 `TITANET_SPEAKER_NEMO_PATH` 가 있을 때 사용한다.
preprocessor 가 상위 `EncDecSpeakerLabelModel` 의 서브모듈이므로 **체크포인트는 1회 restore** 해 두고
encoder 등은 호출하지 않는다(메모리는 .nemo 전체에 상응).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from app.services.speaker_verify.titanet_mel import MelSpectrogramParams, default_mel_params
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_speaker_model: torch.nn.Module | None = None
_preprocessor_device: torch.device | None = None


def nemo_mel_backend_enabled() -> bool:
    b = (settings.titanet_mel_backend or "torchaudio").strip().lower()
    p = (settings.titanet_speaker_nemo_path or "").strip()
    return b in ("nemo", "nemo_preprocessor") and bool(p)


def _app_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _mel_torch_to_numpy_b1ct(mel: torch.Tensor) -> tuple[np.ndarray, int]:
    """preprocessor 출력 → ([1, n_mels, T], n_mels)."""
    x = mel.float().detach().cpu()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(f"unexpected mel rank: {tuple(x.shape)}")
    _, a, b = x.shape
    if a <= b:
        n_mels, tdim = a, b
    else:
        x = x.transpose(1, 2)
        n_mels, tdim = x.shape[1], x.shape[2]
    return x.numpy().astype(np.float32), int(n_mels)


def get_shared_nemo_preprocessor(device: torch.device) -> torch.nn.Module:
    """`.nemo`에서 1회 로드 (preprocessor 참조; 모델 본체는 GC 방지용으로 유지)."""
    global _speaker_model, _preprocessor_device
    with _lock:
        if _speaker_model is None:
            raw = (settings.titanet_speaker_nemo_path or "").strip()
            path = Path(raw) if raw else _app_root() / "models" / "speaker_verification" / "titanet_small_finetuned_final.nemo"
            if not path.is_file():
                raise FileNotFoundError(f"TITANET_SPEAKER_NEMO_PATH 없음 또는 파일 없음: {path}")
            from nemo.collections.asr.models import EncDecSpeakerLabelModel

            # CPU 전용 환경에서 GPU 시도 방지·일관성 (ONNX mel 파이프라인도 CPU)
            use = torch.device("cpu")
            if device.type != "cpu":
                logger.warning(
                    "NeMo mel: 요청 device=%s 이지만 CPU 로 고정합니다 (TitaNet ONNX mel 경로와 동일).",
                    device,
                )

            logger.info("NeMo mel: restore_from 시작 path=%s device=cpu …", path)
            t0 = time.monotonic()
            _speaker_model = EncDecSpeakerLabelModel.restore_from(restore_path=str(path))
            _speaker_model.eval()
            _speaker_model.to(use)
            elapsed = time.monotonic() - t0
            logger.info(
                "NeMo mel: restore_from 완료 elapsed=%.2fs preprocessor 전용 (CPU)",
                elapsed,
            )
        elif _preprocessor_device != device:
            _speaker_model.to(torch.device("cpu"))
        _preprocessor_device = torch.device("cpu")
        return _speaker_model.preprocessor


class NemoTitaNetMelFrontend(nn.Module):
    """ONNX 파이프라인용 NeMo preprocessor mel."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._device = device
        self._prep = get_shared_nemo_preprocessor(device)
        self._n_mels: int = default_mel_params().n_mels

    @property
    def params(self) -> MelSpectrogramParams:
        p = default_mel_params()
        return MelSpectrogramParams(
            sample_rate=p.sample_rate,
            n_fft=p.n_fft,
            n_mels=self._n_mels,
            win_length=p.win_length,
            hop_length=p.hop_length,
            f_min=p.f_min,
            f_max=p.f_max,
        )

    @torch.no_grad()
    def forward(self, samples_1d: np.ndarray) -> np.ndarray:
        if samples_1d.size == 0:
            raise ValueError("empty audio")
        sig = torch.from_numpy(samples_1d.astype(np.float32, copy=False)).unsqueeze(0).to(self._device)
        ln = torch.tensor([sig.shape[1]], device=self._device, dtype=torch.int64)
        mel, _mel_len = self._prep(input_signal=sig, length=ln)
        arr, nm = _mel_torch_to_numpy_b1ct(mel)
        self._n_mels = nm
        return arr


def build_onnx_mel_frontend(device: torch.device, *, log_tag: str) -> nn.Module:
    """ONNX 화자검증용 mel 프론트엔드 (torchaudio | NeMo). NeMo 실패 시 torchaudio 로 폴백."""
    from app.services.speaker_verify.titanet_mel import TitaNetMelFrontend

    if nemo_mel_backend_enabled():
        try:
            fe = NemoTitaNetMelFrontend(device)
            logger.info("[%s] mel 백엔드=nemo path=%s", log_tag, settings.titanet_speaker_nemo_path)
            return fe
        except Exception:
            logger.exception("[%s] NeMo mel 로드 실패 → torchaudio", log_tag)
            return TitaNetMelFrontend(device)
    logger.info("[%s] mel 백엔드=torchaudio", log_tag)
    return TitaNetMelFrontend(device)
