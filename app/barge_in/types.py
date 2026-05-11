"""Barge-in enum 타입."""
from enum import Enum


class BargeInState(str, Enum):
    """Speculative interrupt 상태머신.

    IDLE → SPECULATIVE_INTERRUPT → BARGE_IN_CONFIRMED
                              ↘ (실패/timeout) → IDLE
    """
    IDLE = "idle"
    SPECULATIVE_INTERRUPT = "speculative"
    BARGE_IN_CONFIRMED = "confirmed"


class BargeInRejectReason(str, Enum):
    """Speculative reject 사유 — 로깅/디버깅용."""
    TIMEOUT = "timeout"
    VERIFY_FAILED = "verify_failed"
    VERIFY_ERROR = "verify_error"
    ECHO_GRACE = "echo_grace"
    LOW_ENERGY = "low_energy"
    SHORT_DURATION = "short_duration"
    NOT_ENROLLED = "not_enrolled"
