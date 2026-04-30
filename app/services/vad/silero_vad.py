"""Silero VAD — ML 기반 음성 활동 검출 (2026-04-30 채택, WebRTC 대체).

WebRTC VAD 가 짧은 발화 + 긴 trailing silence 청크에서 `speech_frames / total_frames`
bulk ratio gate (mode=3, threshold=0.5) 로 reject 하던 한계 해결.

근거 사례 (logs/2026-04-30/server_100651.log Turn 4/5):
    "예약은어떻게해요" 발화 0.5~0.8s + trailing 1.3s → speech ratio ~28~38% → reject
    → graph END → silence_check 멘트만 발사 → 통화 무응답 종료.

Silero 는 32ms (=512 샘플 @ 16kHz) frame 단위 per-frame ML 확률을 산출. 청크 안에
음성 frame 이 `silero_min_speech_frames` 이상이면 True. Bulk ratio 와 달리 trailing
silence 길이에 영향 받지 않음.

VADIterator (streaming stateful) 가 아닌 stateless `model(frame, sr)` 직접 호출:
- vad_node 와 _attempt_bargein_verify 둘 다 청크 단위 단발 분류라 streaming state 불필요
- per-call state dict / cleanup 부담 제거
- 청크 길이 ~0.8s (25 frames) ~ 4s (125 frames). frame 당 inference ~200~300μs (CPU,
  ONNX 또는 JIT 단일 thread). 최악 ~40ms — 5초 예산에 무시 가능.

Quality 비교 (Silero 공식 wiki):
    multi-domain ROC-AUC: WebRTC 0.73 → Silero 0.97
    noise vs speech 정확도 (ESC-50): WebRTC ~0% → Silero 87%

알려진 한계 — TTS echo false-positive 는 둘 다 동일 (Azure 합성음이 음성과 음향적
구분 불가). VAD 레이어 교체로 해결 안 됨 — `is_tts_playing` 게이트 + AEC 가 별도 작업.

설치: pip install silero-vad>=6.2
"""
import asyncio

import numpy as np
import torch

from app.services.vad.base import BaseVADService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 동시 통화에서 PyTorch multi-thread inference 가 latency spike + thread contention 유발.
# 공식 wiki 권장 — 모듈 로드 직후 단일 thread 고정.
torch.set_num_threads(1)

# Silero VAD 입력 제약 (v6.2+) — 16kHz 는 정확히 512 샘플 (32ms) frame 만 허용.
# 다른 사이즈는 ValueError. 청크를 이 단위로 슬라이스해 per-frame 추론.
_FRAME_SAMPLES = 512


class SileroVADService(BaseVADService):
    """Silero VAD (v6.2+) 기반 음성 활동 검출 — provider 패턴 호환."""

    name = "silero_vad"

    def __init__(self) -> None:
        self._model = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """모델 lazy load — 첫 detect() 호출 시 1회. 동시 호출 lock 으로 single load."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return

            def _load():
                from silero_vad import load_silero_vad
                return load_silero_vad(onnx=settings.silero_use_onnx)

            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(None, _load)
            self._initialized = True
            logger.info(
                "Silero VAD 모델 로드 완료 onnx=%s threshold=%.2f min_speech_frames=%d",
                settings.silero_use_onnx,
                settings.silero_threshold,
                settings.silero_min_speech_frames,
            )

    async def detect(self, audio_chunk: bytes) -> bool:
        """PCM16 16kHz mono → 발화 여부 bool.

        청크를 512 샘플 frame 으로 슬라이스. 각 frame 의 speech 확률을 모델로 추론.
        p >= silero_threshold 인 frame 이 silero_min_speech_frames 이상이면 True
        (조기 종료 — 임계 도달 즉시 반환).
        """
        if not self._initialized:
            await self.initialize()

        if not audio_chunk:
            return False

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._infer_sync, audio_chunk)

    def _infer_sync(self, audio_chunk: bytes) -> bool:
        # PCM16 → float32 normalized [-1.0, 1.0]
        pcm = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if pcm.size < _FRAME_SAMPLES:
            return False

        # 마지막 자투리는 0-pad 하지 않고 버림 — 32ms 미만 단편은 신호 부족.
        n_frames = pcm.size // _FRAME_SAMPLES
        threshold = settings.silero_threshold
        min_count = settings.silero_min_speech_frames

        speech_count = 0
        with torch.no_grad():
            for i in range(n_frames):
                frame = pcm[i * _FRAME_SAMPLES : (i + 1) * _FRAME_SAMPLES]
                tensor = torch.from_numpy(frame)
                prob = self._model(tensor, 16000).item()
                if prob >= threshold:
                    speech_count += 1
                    if speech_count >= min_count:
                        return True
        return False
