"""Azure TTS PoC — first-byte latency 실측 + 음성 품질 청취 확인.

사용법:
    .env 에 AZURE_SPEECH_KEY / AZURE_SPEECH_REGION 설정 후
    python scripts/azure_tts_poc.py

측정 항목:
    1. SDK 초기화 시간 (1회만)
    2. 짧은 문장 (~30자) 합성 시간 — 5회 평균
    3. 중간 문장 (~70자, greeting 길이) 합성 시간
    4. 긴 문장 (~250자, FAQ 답변 길이) 합성 시간
    5. 출력 bytes 크기 — XTTS 비교용 (μ-law 8kHz, 1초 = 8000 bytes)

산출물:
    scripts/_azure_poc_output_*.ulaw — 청취 검증용 (HxD 등으로 확인)
"""
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.services.tts.azure import AzureTTSService  # noqa: E402


SAMPLES = {
    "short": "안녕하세요. 무엇을 도와드릴까요?",
    "medium": "안녕하세요, 서울중앙병원입니다. 진료 예약과 위치 안내, 진료 시간 문의를 도와드릴 수 있습니다.",
    "long": (
        "서울중앙병원의 정기 건강검진은 평일 오전 아홉 시부터 오후 다섯 시까지 진행되며, "
        "토요일은 오전 아홉 시부터 오후 한 시까지 운영됩니다. 검진 항목은 기본 건강검진과 "
        "종합 건강검진 두 가지 패키지가 있고, 사전 예약이 필수입니다. 보다 자세한 안내가 "
        "필요하시면 진료 시간 내에 다시 전화 주시면 상담원이 연결됩니다."
    ),
}


async def measure(svc: AzureTTSService, label: str, text: str, iters: int = 5) -> None:
    print(f"\n[{label}] text_len={len(text)} chars")
    print(f"  본문: {text[:50]}{'...' if len(text) > 50 else ''}")
    durations = []
    audio_bytes = None
    for i in range(iters):
        t0 = time.perf_counter()
        audio_bytes = await svc.synthesize(text)
        dt = (time.perf_counter() - t0) * 1000
        durations.append(dt)
        print(f"  iter {i + 1}: {dt:>7.1f}ms  bytes={len(audio_bytes):>6d}")
    avg = sum(durations) / len(durations)
    mn = min(durations)
    print(f"  → avg={avg:.1f}ms  min={mn:.1f}ms  bytes/sec={len(audio_bytes) / 8000:.2f}sec재생")
    out = ROOT / f"scripts/_azure_poc_output_{label}.ulaw"
    out.write_bytes(audio_bytes)
    print(f"  saved: {out}")


async def main():
    if not os.getenv("AZURE_SPEECH_KEY") or not os.getenv("AZURE_SPEECH_REGION"):
        print("ERROR: .env 에 AZURE_SPEECH_KEY 와 AZURE_SPEECH_REGION 설정 필요")
        sys.exit(1)

    print("=" * 60)
    print("Azure TTS PoC — first-byte latency 실측")
    print(f"  region={os.getenv('AZURE_SPEECH_REGION')}")
    print(f"  voice={os.getenv('AZURE_TTS_VOICE', 'ko-KR-SunHiNeural (default)')}")
    print("=" * 60)

    t0 = time.perf_counter()
    svc = AzureTTSService()
    await svc.synthesize("초기화")
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"\n[init] SDK 초기화 + 첫 호출 = {init_ms:.1f}ms (워밍업, 비교에서 제외)")

    for label, text in SAMPLES.items():
        await measure(svc, label, text)

    print("\n" + "=" * 60)
    print("XTTS 비교 (이전 측정값, server.log 기반)")
    print("  short  (27 chars)  XTTS:  1,200~2,100ms")
    print("  medium (70 chars)  XTTS:  2,700~4,300ms")
    print("  long   (250+ chars) XTTS: 5,700~12,900ms (5초 하드컷 위반)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
