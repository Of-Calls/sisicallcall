#!/usr/bin/env python3
"""torchaudio `wav_to_mel` vs NeMo `EncDecSpeakerLabelModel.preprocessor` mel 비교 (Step A).

mean diff > 0.1 이면 런타임에서 `TITANET_MEL_BACKEND=nemo` + `TITANET_SPEAKER_NEMO_PATH` 로
학습과 동일 preprocessor 사용을 검토한다.

  python scripts/diagnose_mel.py --compare-nemo
  python scripts/diagnose_mel.py --compare-nemo --wav sample.wav
  python scripts/diagnose_mel.py --compare-nemo --synthetic
  python scripts/diagnose_mel.py --compare-nemo --wav a.wav --onnx app/models/.../x.onnx

Step B (기본 켜짐): 같은 ONNX로 torchaudio mel vs NeMo preprocessor mel 각각 임베딩을 뽑아
코사인 유사도 출력. 0.99 미만이면 mel 불일치가 화자검증/임계값에 영향을 줄 수 있음.

  python scripts/diagnose_mel.py --compare-nemo --no-onnx-emb   # mel 통계만

ONNX 스모크만 할 때:

  python scripts/diagnose_mel.py --onnx path/to.onnx --synthetic
"""
from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("NEMO_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".nemo_cache"))


def _write_synthetic_wav(path: Path, seconds: float = 1.0, sr: int = 16000) -> None:
    n = int(sr * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False, dtype=np.float64)
    sig = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    pcm = np.clip(sig * 32767.0, -32768, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _load_wav_mono_16k(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
        sr = 16000
    return audio, sr


def _to_b80t(m: np.ndarray) -> np.ndarray:
    a = np.asarray(m, dtype=np.float32)
    if a.ndim == 2:
        a = a[np.newaxis, ...]
    if a.shape[1] == 80:
        return a
    if a.shape[2] == 80:
        return np.transpose(a, (0, 2, 1))
    return a


def _synthetic_sine_16k(*, seconds: float = 3.0, hz: float = 200.0) -> np.ndarray:
    """합성 sine (실제 음성 없을 때 mel 차이 관찰용)."""
    n = int(16000 * seconds)
    t = np.arange(n, dtype=np.float32) / 16000.0
    return np.sin(2 * np.pi * hz * t).astype(np.float32)


def _onnx_embedding_from_mel_b1ct(
    sess: Any,
    mel_b1ct: np.ndarray,
    *,
    emb_idx: int,
) -> np.ndarray:
    if mel_b1ct.ndim != 3 or mel_b1ct.shape[0] != 1:
        raise ValueError(f"mel 기대 [1,n_mels,T], got {mel_b1ct.shape}")
    ins = sess.get_inputs()
    audio_name = ins[0].name
    length_name = ins[1].name if len(ins) > 1 else None
    t_frames = int(mel_b1ct.shape[2])
    feeds: dict[str, np.ndarray] = {
        audio_name: np.ascontiguousarray(mel_b1ct, dtype=np.float32),
    }
    if length_name is not None:
        feeds[length_name] = np.array([t_frames], dtype=np.int64)
    out = sess.run(None, feeds)[emb_idx]
    emb = np.asarray(out, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(emb) + 1e-12)
    return (emb / n).astype(np.float32)


def _run_onnx_emb_cosine_torch_vs_nemo_mel(
    mel_torch: np.ndarray,
    mel_nemo: np.ndarray,
    onnx_path: Path,
) -> None:
    """동일 ONNX로 torchaudio mel / NeMo preprocessor mel 각각 임베딩 → 코사인."""
    import onnxruntime as ort

    if not onnx_path.is_file():
        print(f"\n[ONNX emb] skip — file missing: {onnx_path}")
        return

    t = min(int(mel_torch.shape[2]), int(mel_nemo.shape[2]))
    if t < 1:
        print("\n[ONNX emb] skip — empty time axis")
        return
    a = np.ascontiguousarray(mel_torch[:, :, :t], dtype=np.float32)
    b = np.ascontiguousarray(mel_nemo[:, :, :t], dtype=np.float32)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.get_outputs()
    emb_idx = 1 if len(outs) > 1 else 0

    e_t = _onnx_embedding_from_mel_b1ct(sess, a, emb_idx=emb_idx)
    e_n = _onnx_embedding_from_mel_b1ct(sess, b, emb_idx=emb_idx)
    cos = float(np.dot(e_t.astype(np.float64), e_n.astype(np.float64)))

    print()
    print("--- ONNX embedding (same graph, two mels) ---")
    print(f"ONNX: {onnx_path}")
    print(f"aligned T (for both mels): {t}")
    print(
        "cos( ONNX+torchaudio_mel , ONNX+nemo_preprocessor_mel ) = "
        f"{cos:.6f}"
    )
    if cos >= 0.99:
        print("=> cos>=0.99: two mels map to nearly the same embedding direction.")
    else:
        print(
            "=> cos<0.99: mel mismatch can shift speaker scores; "
            "consider TITANET_MEL_BACKEND=nemo + TITANET_SPEAKER_NEMO_PATH."
        )


def _run_compare_nemo(
    *,
    wav: np.ndarray,
    sr: int,
    nemo_path: Path,
    onnx_path: Path | None,
    run_onnx_emb: bool,
) -> None:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")

    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    from app.services.speaker_verify.titanet_mel import wav_to_mel

    if not nemo_path.is_file():
        raise SystemExit(f".nemo 없음: {nemo_path}")

    print(f"restore_from: {nemo_path} (CPU, mel 비교 전용)")
    model = EncDecSpeakerLabelModel.restore_from(restore_path=str(nemo_path))
    model.eval()
    model.to(torch.device("cpu"))

    mel_app = _to_b80t(wav_to_mel(wav, sample_rate=sr))
    print("app wav_to_mel (torchaudio) shape:", mel_app.shape)
    print("app  mean/std:", float(mel_app.mean()), float(mel_app.std()))

    sig = torch.from_numpy(wav).float().unsqueeze(0)
    ln = torch.tensor([wav.shape[0]], dtype=torch.int64)
    with torch.no_grad():
        mel_nemo_t, _ = model.preprocessor(input_signal=sig, length=ln)
    mel_nemo = mel_nemo_t.detach().cpu().numpy()
    mel_nemo = _to_b80t(mel_nemo)
    print("nemo preprocessor shape:", mel_nemo.shape)
    print("nemo mean/std:", float(mel_nemo.mean()), float(mel_nemo.std()))

    ta = min(mel_app.shape[2], mel_nemo.shape[2])
    d = np.abs(mel_app[:, :, :ta] - mel_nemo[:, :, :ta])
    mx, mn = float(d.max()), float(d.mean())
    print(f"aligned T=min(app,nemo)={ta}")
    print(f"max abs diff:  {mx:.6f}")
    print(f"mean abs diff: {mn:.6f}")
    if mn > 0.1:
        print("[결론] mean diff > 0.1 → TITANET_MEL_BACKEND=nemo + TITANET_SPEAKER_NEMO_PATH 권장")
    else:
        print("[결론] mean diff < 0.1 → mel 불일치 가능성 낮음, 임계값·등록 구간·음질 등 다른 원인 검토")

    if run_onnx_emb:
        default_onnx = (
            _ROOT / "app/models/speaker_verification/titanet_small_finetuned_final.onnx"
        )
        path = onnx_path if onnx_path is not None else default_onnx
        _run_onnx_emb_cosine_torch_vs_nemo_mel(mel_app, mel_nemo, path)


def _run_onnx_only(onnx_path: Path, *, synthetic: bool, wav_path: Path | None) -> None:
    import onnxruntime as ort

    from app.services.speaker_verify.titanet_mel import wav_to_mel

    if not onnx_path.is_file():
        raise SystemExit(f"ONNX 없음: {onnx_path}")
    if synthetic:
        tmp = _ROOT / ".tmp_diagnose_mel.wav"
        _write_synthetic_wav(tmp, seconds=3.0)
        wav, sr = _load_wav_mono_16k(tmp)
        tmp.unlink(missing_ok=True)
    else:
        if wav_path is None or not wav_path.is_file():
            raise SystemExit("--wav 또는 --synthetic 필요")
        wav, sr = _load_wav_mono_16k(wav_path)
    mel = _to_b80t(wav_to_mel(wav, sample_rate=sr))
    if mel.shape[2] < 8:
        mel = np.pad(mel, ((0, 0), (0, 0), (0, 8 - mel.shape[2])), mode="constant")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()
    audio_name = ins[0].name
    length_name = ins[1].name if len(ins) > 1 else None
    outs = sess.get_outputs()
    emb_idx = 1 if len(outs) > 1 else 0
    feeds = {audio_name: np.ascontiguousarray(mel, dtype=np.float32)}
    if length_name:
        feeds[length_name] = np.array([mel.shape[2]], dtype=np.int64)
    out = sess.run(None, feeds)[emb_idx]
    emb = np.asarray(out, dtype=np.float32).reshape(-1)
    print("mel shape:", mel.shape, "emb dim:", emb.size, "L2:", float(np.linalg.norm(emb)))


def main() -> None:
    parser = argparse.ArgumentParser(description="mel torchaudio vs NeMo 비교 또는 ONNX 스모크")
    parser.add_argument(
        "--compare-nemo",
        action="store_true",
        help="wav_to_mel vs NeMo preprocessor (Step A)",
    )
    parser.add_argument(
        "--nemo",
        type=Path,
        default=_ROOT / "app/models/speaker_verification/titanet_small_finetuned_final.nemo",
        help="EncDecSpeakerLabelModel .nemo",
    )
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="compare-nemo: 3초 200Hz sine(16k) / ONNX-only: 3초 440Hz WAV",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="ONNX-only 스모크 경로 | compare-nemo 시 임베딩 코사인에 사용(미지정 시 finetuned_final.onnx)",
    )
    parser.add_argument(
        "--no-onnx-emb",
        action="store_true",
        help="compare-nemo: mel 통계만 (ONNX torch vs nemo mel 임베딩 비교 생략)",
    )
    parser.add_argument(
        "--extra-t",
        type=str,
        default="",
        help="ONNX 모드: 추가 mel T 난수 스모크(쉼표) 예 300,600,1200",
    )
    args = parser.parse_args()

    if args.compare_nemo:
        if args.wav is not None and args.wav.is_file():
            wav, sr = _load_wav_mono_16k(args.wav)
            print(f"wav: {args.wav} samples={len(wav)} sr={sr}")
        elif args.synthetic:
            wav = _synthetic_sine_16k(seconds=3.0, hz=200.0)
            sr = 16000
            print(
                "wav: in-memory synthetic 3s 200Hz sine @16kHz "
                "(use --wav for real speech; mel diff clearer than short sine)"
            )
        else:
            wav = _synthetic_sine_16k(seconds=3.0, hz=200.0)
            sr = 16000
            print(
                "wav: in-memory default 3s 200Hz sine @16kHz "
                "(prefer --wav file; --synthetic is the same waveform)"
            )
        _run_compare_nemo(
            wav=wav,
            sr=sr,
            nemo_path=args.nemo,
            onnx_path=args.onnx,
            run_onnx_emb=not args.no_onnx_emb,
        )
        return

    if args.onnx is None:
        raise SystemExit("--compare-nemo 또는 --onnx 중 하나를 지정하세요.")
    _run_onnx_only(args.onnx, synthetic=args.synthetic, wav_path=args.wav)
    if args.extra_t.strip():
        import onnxruntime as ort

        sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        ins = sess.get_inputs()
        audio_name = ins[0].name
        length_name = ins[1].name if len(ins) > 1 else None
        outs = sess.get_outputs()
        emb_idx = 1 if len(outs) > 1 else 0
        for t_s in args.extra_t.split(","):
            t = int(t_s.strip())
            if t < 1:
                continue
            rm = np.random.randn(1, 80, t).astype(np.float32)
            feeds = {audio_name: rm}
            if length_name:
                feeds[length_name] = np.array([t], dtype=np.int64)
            try:
                sess.run(None, feeds)
                print(f"random mel T={t}: OK")
            except Exception as e:
                print(f"random mel T={t}: FAIL {e}")


if __name__ == "__main__":
    main()
