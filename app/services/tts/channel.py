from app.services.tts.base import BaseTTSOutputChannel
from app.utils.config import settings
from app.utils.logger import get_logger

# 모듈 레벨 싱글톤. 노드 · graph runner · call.py 가 import 해서 사용.
# 환경변수 TTS_CHANNEL_MODE 로 구현체 선택.
#   - "mock"   (기본): MockTTSOutputChannel — 테스트 / dev 용, 외부 의존 없음
#   - "twilio"       : TwilioTTSOutputChannel — 프로덕션, WebSocket outbound
# 테스트에서는 monkeypatch 로 이 변수를 Mock 인스턴스로 교체한다.

logger = get_logger(__name__)


def _build_channel() -> BaseTTSOutputChannel:
    mode = (settings.tts_channel_mode or "mock").lower()
    if mode == "twilio":
        from app.services.tts.twilio_channel import TwilioTTSOutputChannel
        logger.info("tts_channel mode=twilio")
        return TwilioTTSOutputChannel()
    from app.services.tts.mock_channel import MockTTSOutputChannel
    logger.info("tts_channel mode=mock")
    return MockTTSOutputChannel()


tts_channel: BaseTTSOutputChannel = _build_channel()
