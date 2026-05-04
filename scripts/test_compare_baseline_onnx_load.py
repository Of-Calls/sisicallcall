"""baseline ONNX(순정 등) 단독 로드 검증 — 레포 루트에서 venv 로 실행.

  venv\\Scripts\\python.exe scripts\\test_compare_baseline_onnx_load.py

종료 코드: 0=로드 성공, 1=실패(load_error 또는 models_ready False), 2=예외.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()

    from app.services.speaker_verify.compare_runtime_bootstrap import (
        bootstrap_speaker_compare_runtime_env_before_imports,
    )

    bootstrap_speaker_compare_runtime_env_before_imports(log=True)

    from app.services.speaker_verify.compare_runtime_env import (
        apply_speaker_compare_runtime_env,
    )

    apply_speaker_compare_runtime_env(log=True)

    from app.services.speaker_verify.titanet_compare import (
        get_titanet_compare_speaker_verify_service,
    )

    svc = get_titanet_compare_speaker_verify_service()
    t0 = time.monotonic()
    try:
        svc.load_baseline_model()
    except Exception as e:
        print("EXCEPTION", type(e).__name__, e)
        return 2
    elapsed = time.monotonic() - t0
    err = svc.load_error
    ready = svc.models_ready
    print("load_error:", repr(err))
    print("models_ready:", ready)
    print("elapsed_sec:", round(elapsed, 2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
