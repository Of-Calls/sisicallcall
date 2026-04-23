"""
TitaNet-S / TitaNet-L → ONNX export 스크립트 (NeMo 환경에서 1회 실행)

실행:
    python scripts/export_titanet.py

출력:
    models/titanet_small.onnx
    models/titanet_large.onnx

NeMo export()는 인코더(신경망) 부분만 ONNX화합니다.
전처리(mel spectrogram)는 추론 시 numpy/librosa로 별도 수행합니다.
"""

import os
import sys

import torch

os.environ.setdefault("NEMO_CACHE_DIR", "C:/torch_cache/nemo")

try:
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
except ImportError:
    print("[ERROR] nemo_toolkit이 설치되어 있지 않습니다.")
    print("  pip install nemo_toolkit[asr]  후 재실행하세요.")
    sys.exit(1)

MODELS = {
    "titanet_small": ("titanet_small", "models/titanet_small.onnx"),
    "titanet_large": ("titanet_large", "models/titanet_large.onnx"),
}

os.makedirs("models", exist_ok=True)


def export(model_name: str, out_path: str) -> None:
    print(f"\n[{model_name}] 로드 중...")
    model = EncDecSpeakerLabelModel.from_pretrained(model_name=model_name)
    model.eval()

    print(f"[{model_name}] ONNX export → {out_path}")
    model.export(
        output=out_path,
        check_trace=True,
    )

    # 입력 메타 정보 저장 (추론 코드에서 참조)
    cfg = model.cfg.preprocessor
    meta = {
        "sample_rate":    int(cfg.sample_rate),
        "n_fft":          int(cfg.n_fft),
        "n_mels":         int(cfg.features),        # mel bin 수
        "win_length":     int(cfg.n_window_size),   # samples
        "hop_length":     int(cfg.n_window_stride),  # samples
        "window":         str(cfg.window),
        "dither":         float(cfg.dither),
        "log_zero_guard": float(cfg.log_zero_guard_value),
    }
    import json
    meta_path = out_path.replace(".onnx", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{model_name}] 전처리 메타 저장 → {meta_path}")
    print(f"[{model_name}] 완료")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    for name, (nemo_id, out_path) in MODELS.items():
        try:
            export(nemo_id, out_path)
        except Exception as e:
            print(f"[ERROR] {name} export 실패: {e}")
            sys.exit(1)

    print("\n모든 모델 export 완료.")
    print("  models/titanet_small.onnx")
    print("  models/titanet_large.onnx")
