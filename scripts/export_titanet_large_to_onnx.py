#!/usr/bin/env python3
"""NeMo 허브/캐시에서 TitaNet Large(`titanet_large`)를 받아 mel 입력 ONNX로 export.

사전 준비: NeMo, PyTorch, onnxruntime(검증), 권장 `onnxscript`(torch.onnx.export 안정화).

  pip install nemo_toolkit["asr"] onnxruntime onnxscript

레포 루트에서:

  python scripts/export_titanet_large_to_onnx.py
  python scripts/export_titanet_large_to_onnx.py --onnx-out models/speech_verification/titanet_large.onnx
  python scripts/export_titanet_large_to_onnx.py --verify-time-frames 300,600,1200 --no-verify

기본 검증 T는 `300,600,1200` 만 포함한다. NeMo `model.export` 폴백 그래프는 T=1201(경계+1)에서 Where
브로드캐스트 오류가 날 수 있어 1201 은 기본에서 제외했다.

입력 계약은 `scripts/export_nemo_to_onnx.py` 와 동일: `audio_signal`=[B,n_mels,T] mel, `length`=[B] int64.
NeMo 버전에 따라 `from_pretrained` 인자 이름이 다를 수 있어 `--pretrained` 로 모델 키를 바꿀 수 있다.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

os.environ.setdefault(
    "NEMO_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".nemo_cache")
)


def _load_pretrained_speaker_model(name: str):
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    # NeMo 버전별 시그니처 호환
    try:
        return EncDecSpeakerLabelModel.from_pretrained(model_name=name)
    except TypeError:
        return EncDecSpeakerLabelModel.from_pretrained(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EncDecSpeakerLabelModel titanet_large → ONNX (mel 입력)",
    )
    parser.add_argument(
        "--pretrained",
        default="titanet_large",
        help="NeMo 프리트레인 식별자 (기본: titanet_large)",
    )
    parser.add_argument(
        "--onnx-out",
        type=Path,
        default=_ROOT / "models" / "speech_verification" / "titanet_large.onnx",
        help="출력 .onnx 경로",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda / cpu (기본: CUDA 가능 시 cuda)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument(
        "--n-mels",
        type=int,
        default=80,
        help="더미 mel 빈 수 (export·검증)",
    )
    parser.add_argument(
        "--t-frames",
        type=int,
        default=600,
        help="torch.onnx.export 더미 mel 시간 프레임 T",
    )
    parser.add_argument(
        "--verify-time-frames",
        type=str,
        default="300,600,1200",
        help="export 후 onnxruntime 검증 T 목록(쉼표). Large+NeMo export 시 1201 제외 권장",
    )
    parser.add_argument(
        "--nemo-export-fallback",
        action="store_true",
        help="torch.onnx.export 대신 NeMo model.export 만 사용",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="export 직후 onnxruntime 다중 T 검증 생략",
    )
    args = parser.parse_args()

    verify_times = tuple(
        int(x.strip()) for x in args.verify_time_frames.split(",") if x.strip()
    )

    print(
        f"from_pretrained({args.pretrained!r}) … "
        "(첫 실행 시 다운로드로 수 분 걸릴 수 있음)"
    )
    model = _load_pretrained_speaker_model(args.pretrained)

    from export_nemo_to_onnx import export_loaded_encdec_speaker_model

    code = export_loaded_encdec_speaker_model(
        model,
        args.onnx_out,
        device_s=args.device,
        opset=args.opset,
        n_mels=args.n_mels,
        t_frames=args.t_frames,
        verify_times=verify_times,
        nemo_export_fallback=args.nemo_export_fallback,
        no_verify=args.no_verify,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
