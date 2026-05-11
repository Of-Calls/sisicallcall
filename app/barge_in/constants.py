"""Barge-in 임계값 / 상수.

모든 수치는 초기값, 실측 후 조정 필요.
"""

# ── Speculative interrupt 진입 ────────────────────────────────────────
TTS_GRACE_PERIOD_MS = 200
"""TTS 송출 시작 후 평가 제외 구간. 초기값, 실측 후 조정.
self-trigger / 잔향 leak 차단 1차 방어선."""

VAD_SPECULATIVE_MIN_MS = 350
"""speech 누적 시간 임계 — 이 값 넘어야 speculative 진입.
초기값, 실측 후 조정. 짧은 기침/'네' 무시용."""

# ── Speaker verification ──────────────────────────────────────────────
VERIFY_MIN_AUDIO_MS = 300
"""verification 시도 최소 candidate 길이.
초기값, 실측 후 조정. TitaNet 짧은 발화 한계 회피."""

# 우리 settings.speaker_verify_threshold (전체 통화 verify 와 동일) 사용 — 별도 override 없음

# ── Echo filter 대안 (우리는 STT interim 없음) ────────────────────────
ENERGY_RMS_MIN_THRESHOLD = 500
"""candidate PCM RMS energy 최소값 — 이 미만은 leak/배경 노이즈로 판단.
초기값, 실측 후 조정. linear16 16kHz 기준."""

# ── Timeout ───────────────────────────────────────────────────────────
SPECULATIVE_TIMEOUT_MS = 800
"""speculative 진입 후 verification 결과 안 오면 자동 reject.
초기값, 실측 후 조정. 무한 정적 방지."""

# ── Rolling buffer ────────────────────────────────────────────────────
ROLLING_BUFFER_MS = 2000
"""verification 입력 prepend 용 — speculative 진입 직전 사용자 음성 보존.
초기값, 실측 후 조정."""

SAMPLE_RATE_HZ = 16000
"""linear16 sample rate — call.py audioop.ratecv 출력과 동일."""

# ── TTS chunk loop ────────────────────────────────────────────────────
TTS_CHUNK_BYTES = 160
"""mulaw 8kHz 20ms — Twilio 권장 단위. call.py _TWILIO_CHUNK_BYTES 와 동일."""

MARK_INTERVAL_CHUNKS = 10
"""200ms (10 chunks × 20ms) 마다 Twilio mark 송출.
초기값, 실측 후 조정. 너무 잦으면 WebSocket 부담."""

# ── Cooldown ──────────────────────────────────────────────────────────
FALSE_SPECULATIVE_RESET_COUNT = 3
"""연속 false speculative N회 누적 시 임계값 상향 (cooldown).
초기값, 실측 후 조정."""

VAD_SPECULATIVE_MIN_MS_COOLDOWN_BUMP = 100
"""cooldown 발동 시 추가되는 ms — VAD_SPECULATIVE_MIN_MS + bump.
초기값, 실측 후 조정."""
