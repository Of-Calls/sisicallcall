"""NeMo baseline 비교 활성 시 프로세스 환경 변수 — `torch` / `onnxruntime` import 전에 적용할 것.

`main.py` 에서는 `load_dotenv` 직후 `compare_runtime_bootstrap`(settings/torch 없음) → `settings` 로드 후
본 모듈의 `apply_speaker_compare_runtime_env` 순으로 호출한다.
"""
from __future__ import annotations

import os

from app.utils.config import settings
from app.utils.logger import get_logger

_logger = get_logger(__name__)


def apply_speaker_compare_runtime_env(*, log: bool = False) -> None:
    if not settings.speaker_verify_compare_enabled:
        return
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if settings.speaker_verify_nemo_force_cpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    if log:
        extra = " + CUDA_VISIBLE_DEVICES=-1" if settings.speaker_verify_nemo_force_cpu else ""
        _logger.info(
            "compare_runtime_env: 화자 비교(이중 ONNX) 프로세스 환경 — "
            "KMP_DUPLICATE_LIB_OK, OMP/MKL/NUMEXPR/NUMBA/OPENBLAS/VECLIB 스레드=1, TOKENIZERS_PARALLELISM=false%s",
            extra,
        )
