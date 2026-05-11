"""Barge-in 모듈 — TTS 송출 중 사용자 발화 감지 + 검증 게이트.

Phase 1~5 (오디오 파이프라인 레벨):
- types: BargeInState enum
- constants: 임계값 (모두 초기값, 실측 후 조정)
- session_state: per-call barge-in 상태 dataclass
- speculative: speculative interrupt 상태머신 (Phase 3)
- verify_gate: speaker verification 백그라운드 (Phase 4)
- rolling_buffer: 2초 분량 PCM 보관 (Phase 4/5)

회귀 안전망: BARGE_IN_ENABLED env 기본 False — true 일 때만 활성.
"""
from app.barge_in.types import BargeInState, BargeInRejectReason

__all__ = ["BargeInState", "BargeInRejectReason"]
