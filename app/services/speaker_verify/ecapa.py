from app.services.speaker_verify.base import BaseSpeakerVerifyService
from app.utils.logger import get_logger

# TODO(대영): ECAPA-TDNN 파인튜닝 완료 후 구현
# 해제 조건: 파인튜닝 완료 보고 후

logger = get_logger(__name__)


class ECAPASpeakerVerifyService(BaseSpeakerVerifyService):
    async def verify(self, audio_chunk: bytes, call_id: str) -> bool:
        # TODO(대영): ECAPA-TDNN 추론 + Redis voiceprint 비교
        raise NotImplementedError

    async def extract_and_store(self, audio_chunk: bytes, call_id: str) -> None:
        # TODO(대영): ECAPA-TDNN 추론 후 Redis call:{call_id}:voiceprint 저장
        duration_ms = int(len(audio_chunk) / (16000 * 2) * 1000)
        logger.info(
            f"[STUB] call_id={call_id} voiceprint 추출 대상 "
            f"audio={duration_ms}ms ({len(audio_chunk)} bytes) "
            f"— ECAPA-TDNN 구현 후 실제 추론 예정"
        )
