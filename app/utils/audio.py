import audioop

from app.utils.logger import get_logger

logger = get_logger(__name__)

# call_id 별 리샘플 상태 — 동시 통화 간 PCM 오염 방지
_RESAMPLE_STATES: dict[str, object] = {}


def mulaw_to_pcm16(mulaw_bytes: bytes, call_id: str = "default") -> bytes:
    """Twilio μ-law 8kHz 오디오를 PCM 16kHz 16-bit로 변환."""
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    pcm_16k, _RESAMPLE_STATES[call_id] = audioop.ratecv(
        pcm_8k, 2, 1, 8000, 16000, _RESAMPLE_STATES.get(call_id)
    )
    return pcm_16k


def reset_resample_state(call_id: str = "default") -> None:
    """통화 종료 시 해당 call_id 리샘플 상태 제거."""
    _RESAMPLE_STATES.pop(call_id, None)
