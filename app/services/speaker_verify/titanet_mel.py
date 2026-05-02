"""PCM16 linear → log-mel [1, 80, T] (NeMo TitaNet 전처리와 동등 목표).

ONNX 백본은 mel 스펙트럼 입력 전제. raw PCM 은 NeMo `get_embedding` 내부와 같이
여기서 mel 로 변환한 뒤 모델에 넣는다 (_extract_embedding 동등 경로).

파라미터: NeMo AudioToMelSpectrogramPreprocessor / titanet-large 계열 기본값 정렬
(sample_rate=16000, n_fft=512, win 25ms, stride 10ms, n_mels=80, log + per_feature norm).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchaudio

_SAMPLE_RATE = 16000
_N_FFT = 512
_WIN_LENGTH = 400  # 25 ms @ 16 kHz
_HOP_LENGTH = 160  # 10 ms
_N_MELS = 80
_F_MAX_HZ = 8000.0  # 음성 대역 (16 kHz 대비 상한)


def pcm16_bytes_to_float_mono(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


class TitaNetMelFrontend(nn.Module):
    """log-mel + 시간축 기준 per-feature 정규화 (NeMo `normalize: per_feature` 근사)."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._device = device
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=_SAMPLE_RATE,
            n_fft=_N_FFT,
            win_length=_WIN_LENGTH,
            hop_length=_HOP_LENGTH,
            n_mels=_N_MELS,
            f_min=0.0,
            f_max=_F_MAX_HZ,
            power=2.0,
            center=True,
        ).to(device)

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
