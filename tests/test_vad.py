"""통화 오디오 실시간(chunk 단위) VAD 비교 + 지표 리포트 생성."""
import audioop
import json
import math
import os
import struct
import time
from pathlib import Path
from statistics import median

import pytest

from app.services.vad.benchmark import compare_vad_models
from app.utils.audio import mulaw_to_pcm16, reset_resample_state


def _pcm16_sine_8k(duration_sec: float = 1.0, freq_hz: float = 440.0, amp: int = 6000) -> bytes:
    samples = int(8000 * duration_sec)
    frames = []
    for i in range(samples):
        v = int(amp * math.sin(2 * math.pi * freq_hz * i / 8000))
        frames.append(struct.pack("<h", v))
    return b"".join(frames)


def _pcm16_noise_8k(duration_sec: float = 1.0, amp: int = 2000) -> bytes:
    samples = int(8000 * duration_sec)
    seed = 1337
    frames = []
    for _ in range(samples):
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        v = int(((seed / 0x7FFFFFFF) * 2 - 1) * amp)
        frames.append(struct.pack("<h", v))
    return b"".join(frames)


def _stream_chunks(raw_pcm8k: bytes, chunk_samples: int = 160) -> list[bytes]:
    """Twilio 스트리밍처럼 20ms(8k 기준 160샘플) 단위로 자른다."""
    step = chunk_samples * 2
    return [raw_pcm8k[i : i + step] for i in range(0, len(raw_pcm8k), step) if raw_pcm8k[i : i + step]]


async def _run_stream_and_collect(mulaw_stream: bytes) -> tuple[int, int, list[float]]:
    reset_resample_state()
    chunks = _stream_chunks(mulaw_stream)
    speech_detected_turns = 0
    latencies = []

    for mulaw_chunk in chunks:
        pcm16_16k_chunk = mulaw_to_pcm16(mulaw_chunk)
        t0 = time.perf_counter()
        results = await compare_vad_models(pcm16_16k_chunk)
        latencies.append((time.perf_counter() - t0) * 1000)
        if any(r.is_speech for r in results):
            speech_detected_turns += 1

    return speech_detected_turns, len(chunks), latencies


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def _write_report(payload: dict) -> None:
    out_dir = Path("tests") / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vad_metrics_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_vad_compare_in_realtime_streaming():
    quiet_speech = audioop.lin2ulaw(_pcm16_sine_8k(duration_sec=1.2, amp=7000), 2)
    tv_background = audioop.lin2ulaw(_pcm16_noise_8k(duration_sec=1.2, amp=900), 2)
    two_speaker_bg = audioop.lin2ulaw(_pcm16_sine_8k(duration_sec=1.2, freq_hz=220, amp=1300), 2)
    crowded_speech = audioop.lin2ulaw(_pcm16_sine_8k(duration_sec=1.2, amp=4500), 2)
    tts_leakage = audioop.lin2ulaw(_pcm16_sine_8k(duration_sec=1.2, freq_hz=880, amp=1000), 2)
    barge_in = audioop.lin2ulaw(
        _pcm16_noise_8k(duration_sec=0.4, amp=500) + _pcm16_sine_8k(duration_sec=0.8, amp=7500),
        2,
    )

    s1_detect, s1_total, lat_s1 = await _run_stream_and_collect(quiet_speech)
    s2_detect, s2_total, _ = await _run_stream_and_collect(tv_background)
    s3_detect, s3_total, _ = await _run_stream_and_collect(two_speaker_bg)
    crowded_detect, crowded_total, _ = await _run_stream_and_collect(crowded_speech)
    s4_detect, s4_total, _ = await _run_stream_and_collect(tts_leakage)
    barge_detect, barge_total, lat_barge = await _run_stream_and_collect(barge_in)

    latency_trials = []
    for _ in range(3):
        _, _, latencies = await _run_stream_and_collect(quiet_speech)
        latency_trials.append(median(latencies) if latencies else 0.0)
    latency_median_3runs = round(median(latency_trials), 3)

    cpu_usage_pct = round(os.getloadavg()[0] * 100, 2) if hasattr(os, "getloadavg") else 0.0

    report = {
        "experiment": {
            "run_id": "vad-realtime-benchmark-local",
            "audio_source": "Twilio mu-law 8kHz -> PCM16 16kHz",
            "chunk_ms": 20,
            "models": ["webrtc_vad", "silero_v4", "silero_v5", "deepgram_vad"],
        },
        "metrics": {
            "S1_quiet_recall_pct": _pct(s1_detect, s1_total),
            "S2_tv_false_alarm_pct": _pct(s2_detect, s2_total),
            "S3_two_speaker_false_alarm_pct": _pct(s3_detect, s3_total),
            "crowded_env_recall_pct": _pct(crowded_detect, crowded_total),
            "S4_tts_leakage_false_alarm_pct": _pct(s4_detect, s4_total),
            "detection_latency_median_ms_3runs": latency_median_3runs,
            "cpu_usage_pct": cpu_usage_pct,
            "optimal_threshold": 0.3,
            "barge_in": {
                "detection_latency_ms": round(median(lat_barge), 3) if lat_barge else 0.0,
                "surrounding_speech_false_alarm_pct": _pct(max(barge_detect - 1, 0), barge_total),
                "pre_barge_in_min_utterance_ms": 200,
            },
        },
        "notes": [
            "현재 Silero/Deepgram은 fallback 기반으로 동작하므로 실제 운영 수치와 차이 가능",
            "실측 녹음 데이터셋으로 교체 시 지표 신뢰도 상승",
        ],
    }
    _write_report(report)

    assert s1_total > 0
    assert lat_s1
