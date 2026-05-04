"""`load_dotenv()` 직후·`app.utils.config` / `torch` import 전에만 호출.

`compare_runtime_env` 는 상단에서 `settings` 를 끌어오므로, 이 모듈은 **os + 로거만** 사용한다
(`get_logger` 경로는 torch 를 끌지 않음).
"""

from __future__ import annotations

import os
from pathlib import Path

from app.utils.logger import get_logger

_logger = get_logger(__name__)


def _truthy_env(name: str) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def bootstrap_speaker_compare_runtime_env_before_imports(*, log: bool = False) -> bool:
    """SPEAKER_VERIFY_COMPARE_ENABLED 가 켜져 있으면 OpenMP/BLAS/numba 스레드 상한을 선설정. 적용 여부 반환."""
    if not _truthy_env("SPEAKER_VERIFY_COMPARE_ENABLED"):
        return False

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if _truthy_env("SPEAKER_VERIFY_NEMO_FORCE_CPU"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    if log:
        numba_hint = ""
        la = (os.environ.get("LOCALAPPDATA") or "").strip()
        if la:
            numba_hint = f" numba_cache_dir={Path(la) / 'numba'}"
        _logger.info(
            "compare_runtime_bootstrap: compare 예정 — 스레드=1 (OMP/MKL/NUMEXPR/NUMBA/OPENBLAS/VECLIB), "
            "KMP_DUPLICATE_LIB_OK, TOKENIZERS_PARALLELISM=false.%s",
            numba_hint,
        )
    return True
