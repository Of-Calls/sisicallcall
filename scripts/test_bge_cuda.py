"""BGE-M3 GPU 활성화 검증 — 모델이 실제로 CUDA에서 돌아가는지 확인.

사용:
    venv/Scripts/python.exe scripts/test_bge_cuda.py
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch


async def main():
    print("=== Torch / CUDA ===")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"VRAM allocated (before): {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    print("\n=== BGE-M3 로드 ===")
    from app.services.embedding.local import BGEM3LocalEmbeddingService
    t0 = time.perf_counter()
    svc = BGEM3LocalEmbeddingService()
    print(f"모델 로드 시간: {time.perf_counter() - t0:.2f}초")

    if torch.cuda.is_available():
        print(f"VRAM allocated (after load): {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    print("\n=== Embed 테스트 ===")
    t0 = time.perf_counter()
    vec = await svc.embed("진료 시간이 어떻게 되나요?")
    print(f"임베딩 시간: {(time.perf_counter() - t0)*1000:.1f}ms")
    print(f"벡터 차원: {len(vec)}")
    print(f"첫 5개 값: {vec[:5]}")


if __name__ == "__main__":
    asyncio.run(main())
