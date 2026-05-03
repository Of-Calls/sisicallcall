#!/usr/bin/env python3
"""소량 WAV(또는 합성 오디오)로 NeMo PyTorch vs ONNX 임베딩을 대조한다.

동일 mel 텐서를 `EncDecSpeakerLabelModel.forward_for_export` 와 ONNX Runtime에 넣어
export 정합성을 확인한다 (앱 런타임과 같은 torchaudio mel: `TitaNetMelFrontend` / 설정값).

사용 (레포 루트, venv 활성화 권장):
  python scripts/compare_nemo_vs_onnx_embeddings.py --synthetic
  python scripts/compare_nemo_vs_onnx_embeddings.py --wav voice/speaker_reference.wav
  python scripts/compare_nemo_vs_onnx_embeddings.py --wav a.wav --wav b.wav \\
    --nemo app/models/speaker_verification/titanet_small_finetuned_final.nemo \\
    --onnx app/models/speaker_verification/titanet_small_finetuned_final.onnx

해석:
  cos( NeMo, ONNX ) 가 0.999 근처면 동일 mel 기준 export·추론이 잘 맞는 편이다.
  0.95 미만이면 mel 파라미터 불일치·다른 체크포인트 export·동적축 이슈 등을 의심한다.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault(
    "NEMO_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".nemo_cache")
)


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(x))
    return x / (n + 1e-12)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_l2(a), _l2(b)))


def _synthetic_16k_mono(sec: float = 1.5, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(sr * sec), dtype=np.float32) / float(sr)
    # 두 주파수 혼합 (너무 단순한 정현파만 피함)
    return (
        0.08 * np.sin(2 * np.pi * 220.0 * t)
        + 0.05 * np.sin(2 * np.pi * 440.0 * t)
        + 0.02 * np.sin(2 * np.pi * 880.0 * t)
    ).astype(np.float32)


def _load_wav_16k_mono(path: Path) -> tuple[np.ndarray, int]:
    from app.services.speaker_verify.titanet_onnx_verifier import (
        load_wav_mono_float32,
        resample_to_16k_mono,
    )

    wav, sr = load_wav_mono_float32(path)
    return resample_to_16k_mono(wav, sr), 16000


def _mel_torchaudio(samples_16k: np.ndarray) -> np.ndarray:
    """앱 ONNX 파이프라인과 동일: `TitaNetMelFrontend` + `TITANET_ONNX_MEL_*` (`.env` / config)."""
    from app.services.speaker_verify.titanet_mel import (
        TitaNetMelFrontend,
        mel_params_from_app_settings,
    )

    dev = torch.device("cpu")
    fe = TitaNetMelFrontend(dev, params=mel_params_from_app_settings())
    return fe(samples_16k.astype(np.float32, copy=False))


def _nemo_embedding_from_mel(
    model,
    mel: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    if mel.ndim != 3 or mel.shape[0] != 1:
        raise ValueError(f"mel shape 기대 [1,n_mels,T], 실제 {mel.shape}")
    mel_t = torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32)).to(device)
    t_frames = int(mel_t.shape[2])
    ln = torch.tensor([t_frames], device=device, dtype=torch.int64)
    with torch.no_grad():
        out = model.forward_for_export(mel_t, ln)
    if isinstance(out, (tuple, list)):
        emb = out[1] if len(out) > 1 else out[0]
    else:
        emb = out
    arr = emb.detach().float().cpu().numpy()
    if arr.ndim >= 2:
        arr = arr.reshape(arr.shape[0], -1)[0]
    return arr.astype(np.float64)


def _onnx_embedding_from_mel(
    sess,
    mel: np.ndarray,
    *,
    in_audio: str,
    in_len: str | None,
    emb_idx: int,
) -> np.ndarray:
    import onnxruntime as ort

    _ = ort  # 타입/린터용
    t_frames = int(mel.shape[2])
    feeds: dict[str, np.ndarray] = {
        in_audio: np.ascontiguousarray(mel, dtype=np.float32),
    }
    if in_len is not None:
        feeds[in_len] = np.array([t_frames], dtype=np.int64)
    outs = sess.run(None, feeds)
    emb = np.asarray(outs[emb_idx], dtype=np.float64)
    if emb.ndim >= 2:
        emb = emb.reshape(emb.shape[0], -1)[0]
    return emb.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeMo forward_for_export vs ONNX 임베딩 대조 (동일 torchaudio mel)"
    )
    parser.add_argument(
        "--nemo",
        type=Path,
        default=_ROOT
        / "app/models/speaker_verification/titanet_small_finetuned_final.nemo",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=_ROOT
        / "app/models/speaker_verification/titanet_small_finetuned_final.onnx",
    )
    parser.add_argument(
        "--wav",
        type=Path,
        action="append",
        default=[],
        help="16kHz로 리샘플 후 비교 (여러 번 지정 가능)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="WAV 없이 1.5초 합성 오디오 1개로 비교",
    )
    parser.add_argument(
        "--max-sec",
        type=float,
        default=4.0,
        help="WAV 앞부분만 사용 (초). mel T 상한 완화",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="NeMo 추론 디바이스 (기본: cuda 가능 시 cuda)",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")

    if not args.nemo.is_file():
        raise SystemExit(f".nemo 없음: {args.nemo}")
    if not args.onnx.is_file():
        raise SystemExit(f".onnx 없음: {args.onnx}")

    wav_jobs: list[tuple[str, np.ndarray]] = []
    if args.synthetic:
        wav_jobs.append(("synthetic_1.5s", _synthetic_16k_mono(1.5)))
    for p in args.wav:
        if not p.is_file():
            raise SystemExit(f"WAV 없음: {p}")
        samples, _sr = _load_wav_16k_mono(p)
        max_n = int(16000 * args.max_sec)
        if samples.size > max_n:
            samples = samples[:max_n].copy()
        wav_jobs.append((str(p), samples))

    if not wav_jobs:
        cand = _ROOT / "voice/speaker_reference.wav"
        if cand.is_file():
            samples, _ = _load_wav_16k_mono(cand)
            max_n = int(16000 * args.max_sec)
            if samples.size > max_n:
                samples = samples[:max_n].copy()
            wav_jobs.append((str(cand), samples))
        else:
            raise SystemExit(
                "비교할 오디오가 없습니다. --synthetic 또는 --wav path.wav 를 지정하세요."
            )

    import onnxruntime as ort
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    print(f"NeMo restore_from: {args.nemo}")
    model = EncDecSpeakerLabelModel.restore_from(restore_path=str(args.nemo))
    model.eval()
    model.to(device)

    print(f"ONNX InferenceSession: {args.onnx}")
    providers = []
    if torch.cuda.is_available():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(str(args.onnx), providers=providers)
    ins = sess.get_inputs()
    in_audio = ins[0].name
    in_len = ins[1].name if len(ins) > 1 else None
    n_out = len(sess.get_outputs())
    emb_idx = 1 if n_out > 1 else 0
    print(f"  ONNX inputs={[i.name for i in ins]} emb_out_idx={emb_idx}")

    print()
    print(
        f"{'sample':<40} {'T_frames':>8} {'cos(NeMo,ONNX)':>14} "
        f"{'L2diff':>12} {'|NeMo|':>10} {'|ONNX|':>10}"
    )
    print("-" * 96)

    for name, samples in wav_jobs:
        mel = _mel_torchaudio(samples)
        t_frames = int(mel.shape[2])
        e_nemo = _nemo_embedding_from_mel(model, mel, device=device)
        e_onnx = _onnx_embedding_from_mel(
            sess, mel, in_audio=in_audio, in_len=in_len, emb_idx=emb_idx
        )
        if e_nemo.shape != e_onnx.shape:
            print(
                f"{name:<40} SHAPE_MISMATCH nemo={e_nemo.shape} onnx={e_onnx.shape}"
            )
            continue
        cos = _cos(e_nemo, e_onnx)
        diff = float(np.linalg.norm(_l2(e_nemo.astype(np.float64)) - _l2(e_onnx.astype(np.float64))))
        print(
            f"{name:<40} {t_frames:>8} {cos:>14.6f} {diff:>12.6e} "
            f"{float(np.linalg.norm(e_nemo)):>10.4f} {float(np.linalg.norm(e_onnx)):>10.4f}"
        )

    print("-" * 96)
    # Windows cp949 콘솔에서 한글 깨짐 방지 — 설명은 docstring(한글) 참고
    print("cos: cosine similarity after L2-normalize each embedding (~1.0 => match).")
    print(
        "mel: torchaudio TitaNetMelFrontend + mel_params_from_app_settings() "
        "(TITANET_ONNX_MEL_* / .env; load_dotenv before run)"
    )
    print(f"NeMo device={device_s}")


if __name__ == "__main__":
    main()
