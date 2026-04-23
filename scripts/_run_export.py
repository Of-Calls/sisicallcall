import os, sys, json, warnings, logging
os.environ["NEMO_CACHE_DIR"] = "C:/torch_cache/nemo"
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel

os.makedirs("models", exist_ok=True)

MODELS = [
    ("titanet_small", "models/titanet_small.onnx"),
    ("titanet_large", "models/titanet_large.onnx"),
]

# titanet_small.onnx 이미 생성됨 — meta만 재저장
SKIP_EXPORT = {"titanet_small"}

def save_meta(model, out_path):
    cfg = model.cfg.preprocessor
    sr = int(cfg.sample_rate)
    meta = {
        "sample_rate": sr,
        "n_fft":       int(cfg.n_fft),
        "n_mels":      int(cfg.features),
        "win_length":  int(cfg.window_size * sr),   # seconds → samples
        "hop_length":  int(cfg.window_stride * sr),  # seconds → samples
        "window":      str(cfg.window),
        "dither":      float(cfg.dither),
        "log_zero_guard": 1e-5,
    }
    meta_path = out_path.replace(".onnx", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta saved: {meta}", flush=True)

for model_name, out_path in MODELS:
    print(f"\n{'='*50}", flush=True)
    print(f"[{model_name}] loading...", flush=True)
    try:
        model = EncDecSpeakerLabelModel.from_pretrained(model_name=model_name)
        model.eval().cpu()
        print(f"[{model_name}] loaded OK", flush=True)

        if model_name in SKIP_EXPORT and os.path.exists(out_path):
            print(f"[{model_name}] ONNX already exists, skipping export", flush=True)
        else:
            print(f"[{model_name}] exporting → {out_path}", flush=True)
            model.export(output=out_path, check_trace=False, verbose=False)

        save_meta(model, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"[{model_name}] DONE ({size_mb:.1f} MB)", flush=True)

    except Exception as e:
        print(f"[ERROR] {model_name}: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc(file=sys.stdout)
        sys.exit(1)

print("\nAll exports complete.", flush=True)
for _, p in MODELS:
    if os.path.exists(p):
        print(f"  {p}  ({os.path.getsize(p)/1e6:.1f} MB)", flush=True)
