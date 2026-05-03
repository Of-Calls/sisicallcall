"""PCM16 linear / WAV float → log-mel [1, n_mels, T] (ONNX TitaNet-Small mel 입력).

ONNX `titanet_small_medium_5epoch_lr5e5.onnx` 는 레포에 **학습 hparams.yaml이 없음**.
mel 스펙은 NeMo `AudioToMelSpectrogramPreprocessor` + TitaNet 관례(16k, 25ms/10ms, 80 mel)와
동일하게 두고, `app.utils.config` (`TITANET_ONNX_MEL_*`) 로 덮어쓴다.

`wav_to_mel` / `TitaNetMelFrontend` 는 동일한 `MelSpectrogramParams` 를 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchaudio

_ONNX_TARGET_SR = 16000


@dataclass(frozen=True)
class MelSpectrogramParams:
    """MelSpectrogram + 이후 log/per-frame norm 까지의 STFT·필터뱅크 입력 파라미터."""

    sample_rate: int
    n_fft: int
    n_mels: int
    win_length: int
    hop_length: int
    f_min: float
    f_max: float


def default_mel_params() -> MelSpectrogramParams:
    """학습 yaml 미수급 시 사용하는 NeMo TitaNet 16kHz 관례값."""
    return MelSpectrogramParams(
        sample_rate=_ONNX_TARGET_SR,
        n_fft=512,
        n_mels=80,
        win_length=400,
        hop_length=160,
        f_min=0.0,
        f_max=8000.0,
    )


def mel_params_from_app_settings() -> MelSpectrogramParams:
    """환경·`.env`에서 ONNX용 mel 파라미터 로드."""
    from app.utils.config import settings

    return MelSpectrogramParams(
        sample_rate=_ONNX_TARGET_SR,
        n_fft=settings.titanet_onnx_mel_n_fft,
        n_mels=settings.titanet_onnx_mel_n_mels,
        win_length=settings.titanet_onnx_mel_win_length,
        hop_length=settings.titanet_onnx_mel_hop_length,
        f_min=settings.titanet_onnx_mel_fmin,
        f_max=settings.titanet_onnx_mel_fmax,
    )


def pcm16_bytes_to_float_mono(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


class TitaNetMelFrontend(nn.Module):
    """log-mel + 시간축 기준 per-feature 정규화 (NeMo `normalize: per_feature` 근사)."""

    def __init__(
        self,
        device: torch.device,
        params: MelSpectrogramParams | None = None,
    ) -> None:
        super().__init__()
        self._device = device
        self._params = params or mel_params_from_app_settings()
        p = self._params
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=p.sample_rate,
            n_fft=p.n_fft,
            win_length=p.win_length,
            hop_length=p.hop_length,
            n_mels=p.n_mels,
            f_min=p.f_min,
            f_max=p.f_max,
            power=2.0,
            center=True,
        ).to(device)

    @property
    def params(self) -> MelSpectrogramParams:
        return self._params

    @torch.no_grad()
    def forward(self, samples_1d: np.ndarray) -> np.ndarray:
        if samples_1d.size == 0:
            raise ValueError("empty audio")
        x = torch.from_numpy(samples_1d).float().to(self._device).unsqueeze(0)
        m = self.mel(x)
        m = torch.log(m.clamp(min=1e-10))
        mean = m.mean(dim=2, keepdim=True)
        std = m.std(dim=2, keepdim=True).clamp(min=1e-5)
        m = (m - mean) / std
        return m.cpu().numpy().astype(np.float32)


def wav_to_mel(
    wav: np.ndarray,
    *,
    sample_rate: int,
    device: torch.device | None = None,
    params: MelSpectrogramParams | None = None,
) -> np.ndarray:
    """1D float32 WAV → log-mel `[1, n_mels, T]` (`TitaNetMelFrontend.forward` 와 동일).

    `sample_rate` 가 mel 설정의 `sample_rate`(기본 16k)와 다르면 torchaudio 로 리샘플한다.
    """
    dev = device or torch.device("cpu")
    p = params or mel_params_from_app_settings()
    if wav.ndim != 1:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    else:
        wav = wav.astype(np.float32, copy=False)

    if sample_rate != p.sample_rate:
        t = torch.from_numpy(wav).unsqueeze(0)
        wav = torchaudio.functional.resample(
            t, orig_freq=sample_rate, new_freq=p.sample_rate
        ).squeeze(0).numpy()

    fe = TitaNetMelFrontend(dev, params=p)
    return fe(wav)
