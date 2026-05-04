#!/usr/bin/env python3
"""models/speech_verification/*.nemo → 동일 폴더 .onnx 일괄 변환 (재실행용).

  레포 루트: venv\\Scripts\\python.exe scripts\\export_speech_verification_onnx_batch.py

내부적으로 `scripts/export_nemo_to_onnx.py` 를 호출한다.
검증은 T=1201 경계에서 NeMo model.export 그래프가 깨질 수 있어 300,600,1200 만 통과시킨다
(앱 `TITANET_FINETUNED_ONNX_MAX_MEL_FRAMES` 와 정합).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_EXPORT = _ROOT / "scripts" / "export_nemo_to_onnx.py"
_VERIFY = "300,600,1200"

_JOBS: tuple[tuple[str, str], ...] = (
    (
        "models/speech_verification/titanet_small_finetuned_final.nemo",
        "models/speech_verification/titanet_small_finetuned_final.onnx",
    ),
    (
        "models/speech_verification/titanet-s.nemo",
        "models/speech_verification/titanet-s.onnx",
    ),
)


def main() -> int:
    if not _EXPORT.is_file():
        print("missing", _EXPORT, file=sys.stderr)
        return 2
    exe = sys.executable
    for nemo_rel, onnx_rel in _JOBS:
        nemo = _ROOT / nemo_rel
        onnx = _ROOT / onnx_rel
        if not nemo.is_file():
            print(f"skip (no .nemo): {nemo}", file=sys.stderr)
            continue
        cmd = [
            exe,
            str(_EXPORT),
            "--nemo-in",
            str(nemo),
            "--onnx-out",
            str(onnx),
            "--verify-time-frames",
            _VERIFY,
        ]
        print("+", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
