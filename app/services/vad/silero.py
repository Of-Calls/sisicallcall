import time

import numpy as np
import torch

from app.services.vad.base import BaseVADService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SAMPLE_RATE = 16000
# torch hub 캐시를 ASCII 경로로 강제 지정 (한글 사용자명 경로에서 JIT 로드 실패 방지)
_HUB_DIR = "C:/torch_cache"
torch.hub.set_dir(_HUB_DIR)


class SileroVADService(BaseVADService):
    def __init__(self) -> None:
        logger.info(f"Silero VAD 모델 로드 중... (hub_dir={_HUB_DIR})")
        try:
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model.eval()
            logger.info("Silero VAD 모델 로드 완료")
        except Exception as e:
            logger.error(f"Silero VAD 모델 로드 실패: {e}")
            raise

    async def detect(self, audio_chunk: bytes) -> dict:
        """PCM 16kHz 16-bit mono bytes → VAD 결과 dict 반환.

        Silero VAD는 정확히 512 samples 단위로만 추론 가능.
        입력을 512-sample 윈도우로 쪼개 각각 추론한 뒤 최대 score를 반환.

        Returns:
            {
                "is_speech": bool,
                "score": float,   # 윈도우별 최대 score
                "duration_ms": int,
            }
        """
        _WINDOW = 512  # Silero VAD 고정 요구 사항 (16kHz 기준)

        t_start = time.monotonic()
        try:
            samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            duration_ms = int(len(samples) / _SAMPLE_RATE * 1000)

            if len(samples) < _WINDOW:
                logger.debug(f"VAD 스킵: 청크 너무 짧음 ({len(samples)} samples)")
                return {"is_speech": False, "score": 0.0, "duration_ms": duration_ms}

            # 512-sample 윈도우로 분할 (나머지 버림)
            n_windows = len(samples) // _WINDOW
            windows = samples[: n_windows * _WINDOW].reshape(n_windows, _WINDOW)
            batch = torch.from_numpy(windows)

            with torch.no_grad():
                scores = self._model(batch, _SAMPLE_RATE)  # shape: (n_windows,)

            max_score: float = scores.max().item()
            is_speech = max_score >= settings.vad_threshold
            return {"is_speech": is_speech, "score": round(max_score, 4), "duration_ms": duration_ms}

        except Exception as e:
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            logger.warning(f"VAD 추론 실패 (elapsed={elapsed_ms}ms): {e}")
            return {"is_speech": False, "score": 0.0, "duration_ms": 0}
