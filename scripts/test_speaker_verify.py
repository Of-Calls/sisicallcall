"""TitaNet 화자검증 통합 테스트.

테스트 케이스:
    1. 모델 로딩 확인
    2. voiceprint 등록 (extract_and_store)
    3. 동일 목소리 검증 (verify -> True 예상)
    4. bypass 동작 (미등록 시 True 반환)
    5. cleanup 후 bypass 복원

사용:
    venv/Scripts/python.exe scripts/test_speaker_verify.py
"""
import asyncio
import os
import sys
import time
import wave

import numpy as np
import scipy.signal as sps

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


def load_wav_as_pcm(path: str, max_sec: float = 5.0, target_sr: int = 16000) -> bytes:
    """WAV 파일을 16kHz 16-bit PCM bytes로 읽기 (리샘플링 포함)."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n_frames = min(int(sr * max_sec), w.getnframes())
        raw = w.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    if sr != target_sr:
        n_target = int(len(samples) * target_sr / sr)
        samples = sps.resample(samples, n_target)

    return samples.astype(np.int16).tobytes()


async def main():
    ref_path = "voice/speaker_reference.wav"
    if not os.path.exists(ref_path):
        print(f"ERROR: {ref_path} 없음")
        return

    print("=" * 55)
    print("TitaNet 화자검증 테스트")
    print("=" * 55)

    # Test 1: 모델 로딩
    print("\n[1] TitaNet 모델 로딩...")
    t0 = time.perf_counter()
    from app.services.speaker_verify.titanet import TitaNetSpeakerVerifyService
    svc = TitaNetSpeakerVerifyService()
    print(f"    로딩 완료: {time.perf_counter() - t0:.2f}초")

    call_id = "test_call_001"
    audio_5s = load_wav_as_pcm(ref_path, max_sec=5.0)
    audio_2s = load_wav_as_pcm(ref_path, max_sec=2.0)
    print(f"    enrollment: {len(audio_5s)} bytes ({len(audio_5s)/32000:.1f}s)")
    print(f"    verify:     {len(audio_2s)} bytes ({len(audio_2s)/32000:.1f}s)")

    # Test 2: voiceprint 등록
    print("\n[2] voiceprint 등록...")
    t0 = time.perf_counter()
    await svc.extract_and_store(audio_5s, call_id)
    print(f"    등록 완료: {time.perf_counter() - t0:.2f}초")

    # Test 3: 동일 목소리 verify
    print("\n[3] 동일 목소리 검증...")
    t0 = time.perf_counter()
    is_verified, similarity = await svc.verify(audio_2s, call_id)
    elapsed = (time.perf_counter() - t0) * 1000
    result = "PASS" if is_verified else "FAIL (threshold 조정 필요)"
    print(f"    similarity: {similarity:.4f}")
    print(f"    verified:   {is_verified} [{result}]")
    print(f"    latency:    {elapsed:.1f}ms")

    # Test 4: bypass (미등록 call_id)
    print("\n[4] bypass 동작 (미등록 call_id)...")
    is_bypass, sim_bypass = await svc.verify(audio_2s, "unknown_call")
    result4 = "PASS" if is_bypass else "FAIL"
    print(f"    verified:   {is_bypass} (True 예상) [{result4}]")

    # Test 5: cleanup 후 bypass
    print("\n[5] cleanup 후 bypass 복원...")
    svc.cleanup(call_id)
    is_after, _ = await svc.verify(audio_2s, call_id)
    result5 = "PASS" if is_after else "FAIL"
    print(f"    cleanup 후: {is_after} (True 예상) [{result5}]")

    print("\n" + "=" * 55)
    print("테스트 완료")
    print(f"  similarity={similarity:.4f}  threshold=0.40")
    if similarity < 0.40:
        print("  [주의] similarity 낮음 - threshold 조정 검토 필요")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
