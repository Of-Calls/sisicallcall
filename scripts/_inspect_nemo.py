"""titanet_small .nemo 파일 다운로드 후 state_dict 키 출력 (1회용 검사 스크립트)."""
import io
import json
import os
import sys
import zipfile

import requests
import torch

SMALL_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/"
    "titanet_small/versions/v1/files/titanet-small.nemo"
)

OUT = "models/_titanet_small_inspect.nemo"
os.makedirs("models", exist_ok=True)

if not os.path.exists(OUT):
    print("Downloading titanet-small.nemo ...")
    r = requests.get(SMALL_URL, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done = 0
    with open(OUT, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"  {done/1e6:.1f} / {total/1e6:.1f} MB", end="\r")
    print("\nDownload complete.")
else:
    print(f"Using cached: {OUT}")

print("\n=== ZIP contents ===")
with zipfile.ZipFile(OUT) as z:
    for name in z.namelist():
        print(" ", name)

    # config
    yaml_files = [n for n in z.namelist() if n.endswith(".yaml")]
    if yaml_files:
        print(f"\n=== {yaml_files[0]} ===")
        print(z.read(yaml_files[0]).decode())

    # weights
    ckpt_files = [n for n in z.namelist() if n.endswith(".ckpt") or n.endswith(".pt")]
    if ckpt_files:
        print(f"\n=== state_dict keys: {ckpt_files[0]} ===")
        data = io.BytesIO(z.read(ckpt_files[0]))
        ckpt = torch.load(data, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        for k, v in sd.items():
            print(f"  {k:70s}  {tuple(v.shape)}")
