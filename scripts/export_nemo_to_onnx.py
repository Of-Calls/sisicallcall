#!/usr/bin/env python3
"""개발자 전용: EncDecSpeakerLabelModel `.nemo` → ONNX.

**앱 통화 런타임의 화자 임베딩은 ONNX Runtime만** 사용한다 (EncDecSpeakerLabelModel 전체를 통화 경로에 올리지 않음).
mel 은 기본 `torchaudio` + `TITANET_ONNX_MEL_*` (`TitaNetMelFrontend`)이며, 불일치 시
`TITANET_MEL_BACKEND=nemo` + `TITANET_SPEAKER_NEMO_PATH` 로 NeMo preprocessor 만 쓸 수 있다(임베딩은 여전히 ONNX).

이 스크립트는 아티팩트를 새로 만들거나 재export할 때만 쓴다.

NeMo `EncDecSpeakerLabelModel.forward_for_export(audio_signal, length)` 는 **전처리 없이**
mel 스펙트로그를 받는다 (서비스 ONNX 계약과 동일).

기본 입출력:
  입력:  app/models/speaker_verification/titanet_small_finetuned_final.nemo
  출력:  app/models/speaker_verification/titanet_small_finetuned_final.onnx

기본은 `torch.onnx.export`(더미 mel T=300, `dynamic_axes`) 후 T=300/600 onnxruntime 검증
(NeMo `model.export` 폴백 그래프는 종종 내부 시간축이 ~600으로 박혀 T=1200에서 Where 오류가 난다).
`pip install onnxscript` 가 되어 있으면 torch 경로 성공 확률이 높다. 실패 시 NeMo `model.export` 로 폴백한다.

사용 예 (레포 루트):
  python scripts/export_nemo_to_onnx.py
  python scripts/export_nemo_to_onnx.py --nemo-in path/to/model.nemo --onnx-out out.onnx
  python scripts/export_nemo_to_onnx.py --onnx-out out.onnx --verify-only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 레포 루트를 path에 넣는다 (이 스크립트만 NeMo import; 앱 런타임과 무관).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# NeMo import 전 캐시(선택)
os.environ.setdefault(
    "NEMO_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".nemo_cache")
)


def _default_mel_batch(device: "torch.device", n_mels: int = 80, t_frames: int = 300):
    import torch

    mel = torch.randn(1, n_mels, t_frames, device=device, dtype=torch.float32)
    length = torch.tensor([t_frames], device=device, dtype=torch.int64)
    return (mel, length)


def export_torch_onnx_export(
    model,
    onnx_out: Path,
    *,
    device: "torch.device",
    opset: int,
    n_mels: int,
    t_frames: int,
) -> None:
    """`torch.onnx.export` 로 mel 입력 동적 T ONNX 생성 (NeMo `model.export` 와 별개)."""
    import torch
    import torch.nn as nn

    class _MelEncoderDecoder(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, audio_signal, length):
            return self.m.forward_for_export(audio_signal, length)

    wrap = _MelEncoderDecoder(model).eval().to(device)
    dummy_mel = torch.randn(1, n_mels, t_frames, device=device, dtype=torch.float32)
    dummy_len = torch.tensor([t_frames], device=device, dtype=torch.int64)
    torch.onnx.export(
        wrap,
        (dummy_mel, dummy_len),
        str(onnx_out),
        opset_version=opset,
        input_names=["audio_signal", "length"],
        output_names=["logits", "embs"],
        dynamic_axes=_ONNX_DYNAMIC_AXES_FULL,
        do_constant_folding=True,
    )


# EncDecSpeakerLabelModel.forward_for_export(audio_signal, length) 계약과 맞춘다.
# 출력 이름은 NeMo/ONNX 버전에 따라 다를 수 있어 export 실패 시 입력만 동적 축으로 재시도.
_ONNX_DYNAMIC_AXES_FULL = {
    "audio_signal": {0: "batch", 2: "time"},
    "length": {0: "batch"},
    "logits": {0: "batch"},
    "embs": {0: "batch"},
}
_ONNX_DYNAMIC_AXES_INPUTS_ONLY = {
    "audio_signal": {0: "batch", 2: "time"},
    "length": {0: "batch"},
}


def verify_onnx_mel_time_axis(
    onnx_path: Path,
    *,
    n_mels: int = 80,
    time_frames: tuple[int, ...] = (300, 600),
) -> bool:
    """재export 한 ONNX가 서로 다른 mel 시간 길이 T에서 shape 오류 없이 돌아가는지 확인.

    Returns:
        True: 모든 T에서 sess.run 성공. False: 한 T라도 실패(그래프가 여전히 고정 T에 맞춰짐).
    """
    import numpy as np
    import onnxruntime as ort

    if not onnx_path.is_file():
        raise SystemExit(f"ONNX 파일 없음: {onnx_path}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()
    if not ins:
        raise SystemExit("ONNX 입력 메타 없음")
    audio_name = ins[0].name
    length_name = ins[1].name if len(ins) > 1 else None

    print(f"verify: {onnx_path} inputs={[i.name for i in ins]}")
    for T in time_frames:
        mel = np.random.randn(1, n_mels, T).astype(np.float32)
        feeds: dict[str, np.ndarray] = {audio_name: mel}
        if length_name is not None:
            feeds[length_name] = np.array([T], dtype=np.int64)
        try:
            sess.run(None, feeds)
        except Exception as e:
            print(f"  FAIL mel time T={T}: {e}")
            print(
                "\n[검증 실패] .onnx 파일은 디스크에 있으나, 위 T에서 onnxruntime 추론이 깨집니다. "
                "NeMo `model.export` 폴백은 종종 내부 시간축이 trace T(또는 배수)로 박혀 "
                "`Where` 브로드캐스트(예: 600×1200)가 납니다. 대응: (1) `pip install onnxscript` 후 "
                "재export해 `torch.onnx.export` 경로를 쓰기 (2) `--verify-time-frames 300,600` 처럼 "
                "그래프가 견디는 T만 검증 (3) 런타임에서 mel T가 그 한도 이하가 되게 PCM 길이 제한 "
                "(4) `--no-verify` 는 미검증 산출물용.\n"
            )
            return False
        print(f"  OK mel time T={T}")
    print(f"verify 전부 통과: T={', '.join(str(t) for t in time_frames)}.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="NeMo speaker .nemo → ONNX")
    parser.add_argument(
        "--nemo-in",
        type=Path,
        default=_ROOT
        / "app/models/speaker_verification/titanet_small_finetuned_final.nemo",
        help="입력 .nemo 경로",
    )
    parser.add_argument(
        "--onnx-out",
        type=Path,
        default=_ROOT
        / "app/models/speaker_verification/titanet_small_finetuned_final.onnx",
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
        help="더미 입력 mel 빈(feat) — export 실패 시 fallback trace 용",
    )
    parser.add_argument(
        "--t-frames",
        type=int,
        default=600,
        help="torch.onnx.export 더미 mel 시간 프레임 T (기본 600≈6s; 너무 작으면 고정 T trace 위험)",
    )
    parser.add_argument(
        "--verify-time-frames",
        type=str,
        default="300,600,1200",
        help="export 후 onnxruntime 검증 T 목록(쉼표). 통화 12s급 mel≈1200 — 여기서 실패하면 재export 필요",
    )
    parser.add_argument(
        "--nemo-export-fallback",
        action="store_true",
        help="torch.onnx.export 대신 기존 NeMo model.export 만 사용",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="export 직후 onnxruntime 다중 T 검증 생략",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=".nemo 없이 기존 ONNX만 검증",
    )
    args = parser.parse_args()

    verify_times = tuple(
        int(x.strip()) for x in args.verify_time_frames.split(",") if x.strip()
    )

    if args.verify_only:
        ok = verify_onnx_mel_time_axis(
            args.onnx_out, n_mels=args.n_mels, time_frames=verify_times
        )
        raise SystemExit(0 if ok else 1)

    if not args.nemo_in.is_file():
        raise SystemExit(f".nemo 파일 없음: {args.nemo_in}")

    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    print(f"restore_from: {args.nemo_in}")
    model = EncDecSpeakerLabelModel.restore_from(restore_path=str(args.nemo_in))
    model.eval()
    model.to(device)

    args.onnx_out.parent.mkdir(parents=True, exist_ok=True)

    input_example = None
    for kwargs in (
        {"max_batch": 1, "max_dim": 256},
        {"max_batch": 1, "max_dim": 128},
        {},
    ):
        if not hasattr(model, "input_example"):
            break
        try:
            input_example = (
                model.input_example(**kwargs) if kwargs else model.input_example()
            )
        except TypeError:
            continue
        except Exception:
            input_example = None
            continue
        if input_example is not None:
            if isinstance(input_example, tuple):
                input_example = tuple(
                    x.to(device) if hasattr(x, "to") else x for x in input_example
                )
            print(f"input_example OK kwargs={kwargs or 'none'}")
            break

    if input_example is None:
        print(
            "input_example 없음/실패 - NeMo fallback 시 더미 mel [1, n_mels, T] 사용 "
            f"(n_mels={args.n_mels}, T={args.t_frames})"
        )
        input_example = _default_mel_batch(device, args.n_mels, args.t_frames)

    print(f"export → {args.onnx_out} (device={device_s}, opset={args.opset})")
    exported = False
    with torch.no_grad():
        if not args.nemo_export_fallback:
            try:
                export_torch_onnx_export(
                    model,
                    args.onnx_out,
                    device=device,
                    opset=args.opset,
                    n_mels=args.n_mels,
                    t_frames=args.t_frames,
                )
                print(
                    "torch.onnx.export 완료 (dynamic_axes: audio_signal time + length batch)."
                )
                exported = True
            except Exception as e:
                err = str(e).lower()
                hint = ""
                if "onnxscript" in err:
                    hint = "  → `pip install onnxscript` 후 재시도하면 torch 경로가 살아날 수 있습니다.\n"
                print(
                    f"torch.onnx.export 실패: {e}\n{hint}  → NeMo model.export 로 재시도합니다."
                )

        if not exported:
            try:
                model.export(
                    str(args.onnx_out),
                    input_example=input_example,
                    onnx_opset_version=args.opset,
                    check_trace=False,
                    dynamic_axes=_ONNX_DYNAMIC_AXES_FULL,
                )
            except Exception as e:
                print(
                    f"dynamic_axes(입력+출력) NeMo export 실패: {e}\n"
                    "  → 입력 축만 동적으로 재시도합니다."
                )
                model.export(
                    str(args.onnx_out),
                    input_example=input_example,
                    onnx_opset_version=args.opset,
                    check_trace=False,
                    dynamic_axes=_ONNX_DYNAMIC_AXES_INPUTS_ONLY,
                )
            print("NeMo model.export 완료.")
    print("export 완료 (파일 쓰기 끝).")
    if not args.no_verify:
        ok = verify_onnx_mel_time_axis(
            args.onnx_out, n_mels=args.n_mels, time_frames=verify_times
        )
        if not ok:
            print(
                "종료 코드 1: export 산출물은 있으나 검증 실패. "
                "통과하는 T 범위의 ONNX로 교체하거나 onnxscript 설치 후 torch 재export를 권장합니다."
            )
            raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
