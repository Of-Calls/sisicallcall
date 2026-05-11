"""Per-call barge-in 상태.

call.py 의 nonlocal 변수들과 분리해서 관리 — barge-in 관련 state 만 묶음.
기존 is_speaking / audio_buffer 등 nonlocal 은 점진적 대체 (한 번에 X).
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from app.barge_in.types import BargeInState


@dataclass
class BargeInSessionState:
    # ── Turn 관리 ────────────────────────────────────────────────────
    active_turn_id: int = 0
    """barge-in confirmed 시 +1. stale 검사 기준."""

    # ── Task 추적 (Phase 2/3에서 주입) ────────────────────────────────
    active_tts_task: asyncio.Task | None = None
    active_filler_task: asyncio.Task | None = None
    active_graph_task: asyncio.Task | None = None
    active_graph_turn_id: int | None = None
    # Step 3 — TTS 재생 background task (main loop block 회피용)
    active_play_task: asyncio.Task | None = None

    # ── Barge-in 상태 ────────────────────────────────────────────────
    barge_in_state: BargeInState = BargeInState.IDLE
    tts_paused: bool = False
    tts_resume_event: asyncio.Event = field(default_factory=asyncio.Event)
    speculative_started_at: float = 0.0

    # ── TTS 추적 ─────────────────────────────────────────────────────
    tts_started_at: float = 0.0
    last_bot_utterance: str = ""
    active_tts_chunk_index: int = 0
    """현재 송출된 chunk 번호 (Phase 5 mark 기반 spoken portion 계산용)."""

    # ── Candidate / rolling (Phase 4/5에서 활용) ──────────────────────
    candidate_pcm: bytearray = field(default_factory=bytearray)
    """speculative 진입 후 누적되는 사용자 PCM (verification 입력)."""

    rolling_pcm_buffer: Any = None
    """RollingPCMBuffer 인스턴스 (Phase 5에서 주입)."""

    # ── Twilio mark (Phase 5) ─────────────────────────────────────────
    last_sent_mark: str | None = None
    last_played_mark: str | None = None
    last_played_chunk_index: int = 0

    # ── Speech 누적 추적 ─────────────────────────────────────────────
    speech_active: bool = False
    speech_started_at: float = 0.0

    # ── Cooldown ─────────────────────────────────────────────────────
    false_speculative_count: int = 0
    recent_rejected_embeddings: deque = field(
        default_factory=lambda: deque(maxlen=5)
    )

    # ── Step 2 — inline barge-in verify (TTS 송출 중) ─────────────────
    # batch 송출 + media event 차단 우회 + RMS pre-gate + 0.8s 누적 + VAD + TitaNet
    bargein_pcm_buffer: bytearray = field(default_factory=bytearray)
    bargein_ratecv_state: Any = None
    bargein_silence_after_speech: int = 0
    bargein_diag_count: int = 0
    bargein_verify_inflight: bool = False  # 동시 verify 호출 방지

    def reset_for_new_turn(self) -> None:
        """새 turn 시작 시 호출 — barge-in state 만 리셋, turn_id 는 유지."""
        self.barge_in_state = BargeInState.IDLE
        self.tts_paused = False
        self.tts_resume_event.clear()
        self.candidate_pcm.clear()
        self.speech_active = False
        self.speech_started_at = 0.0
        self.active_tts_chunk_index = 0
        self.last_sent_mark = None
        self.last_played_mark = None
        self.last_played_chunk_index = 0
