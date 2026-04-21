from app.agents.conversational.state import CallState
from app.services.vad.benchmark import compare_vad_models
from app.utils.logger import get_logger

# TODO(주미): VAD threshold 연구 완료 후 구현 — architecture.md 참조
# 해제 조건: 업종별 최적 threshold 도출 후 팀장 보고

logger = get_logger(__name__)


async def vad_node(state: CallState) -> dict:
    results = await compare_vad_models(state["audio_chunk"])
    speech_votes = sum(1 for result in results if result.is_speech)
    is_speech = speech_votes >= 2

    # 판정/지연/가용성 정보를 로그로 남긴다.
    logger.info(
        "VAD 앙상블 결과 call_id=%s turn=%s 음성판정수=%s/%s 상세=%s",
        state["call_id"],
        state["turn_index"],
        speech_votes,
        len(results),
        [
            {
                "모델": r.model,
                "음성여부": r.is_speech,
                "지연ms": r.latency_ms,
                "사용가능": r.available,
                "비고": r.note,
            }
            for r in results
        ],
    )
    return {"is_speech": is_speech}