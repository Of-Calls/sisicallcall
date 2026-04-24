import audioop
import base64
import json
import time

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.agents.conversational.graph import build_call_graph
from app.agents.conversational.state import CallState
from app.api.v1._tenant_helpers import (
    get_greeting,
    get_tenant_name,
    resolve_tenant_id,
)
from app.core.events import CALL_ENDED, CALL_STARTED
from app.services.session.redis_session import RedisSessionService
from app.services.stt.deepgram_streaming import DeepgramStreamingSTTService
from app.services.tts.channel import tts_channel
from app.utils.audio import mulaw_to_pcm16, reset_resample_state
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 그래프 싱글톤 (앱 기동 시 1회 컴파일)
_graph = build_call_graph()
_session_service = RedisSessionService()
_streaming_stt = DeepgramStreamingSTTService()

# 발화 감지 파라미터 — Silero VAD(주미) 구현 전 임시 에너지 기반 묵음 감지
_SPEECH_RMS_THRESHOLD = 1200   # RMS 임계값 (묵음 vs 발화) — TTS 에코 필터링 목적으로 800→1200 상향
_SILENCE_CHUNKS_TO_END = 40    # 발화 종료 판단 묵음 청크 수 (~800ms, 청크당 ~20ms)
_MIN_UTTERANCE_BYTES = 16000   # 최소 발화 길이 (~500ms at 16kHz 16-bit)
_MAX_UTTERANCE_BYTES = 320000  # 최대 발화 길이 (~10s)

# 침묵 감지 파라미터
_SILENCE_FIRST_SEC = 10.0      # 첫 번째 침묵 알림 기준 (초)
_SILENCE_SECOND_SEC = 10.0     # 첫 알림 후 추가 대기 → escalation (초)
# TTS 재생 예상 속도 — 한국어 평균 발화 속도 ≈ 5자/초 (보수적)
# ainvoke 완료 후 response_text 길이로 재생 시간 추정, 침묵 타이머 유예에 사용
_KO_TTS_CHARS_PER_SEC = 5.0
# greeting 재생 완료까지 타이머를 유예하는 버퍼 (greeting은 보통 8~12초 음성)
# emit 시점 기준이라 재생 완료 전에 침묵 알림이 울리는 것을 방지
_GREETING_PLAY_BUFFER_SEC = 13.0
_MSG_SILENCE_CHECK = "통화 중이십니까? 불편한 점이 있으시면 말씀해 주세요."
_MSG_SILENCE_ESCALATION = "전화 연결이 원활하지 않은 것 같습니다. 상담원에게 연결해 드리겠습니다."


@router.post("/incoming")
async def incoming_call(request: Request):
    """
    Twilio가 전화 수신 시 호출하는 webhook.
    TwiML을 반환해 Twilio Media Streams WebSocket 연결을 지시한다.
    """
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    to_number = form.get("To", "unknown")
    tenant_id = await resolve_tenant_id(to_number)

    logger.info(f"[{CALL_STARTED}] call_sid={call_sid} to={to_number} tenant_id={tenant_id}")

    ws_url = f"wss://{settings.base_url.removeprefix('https://').removeprefix('http://')}/call/ws/{call_sid}"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}">
      <Parameter name="tenant_id" value="{tenant_id}"/>
    </Stream>
  </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@router.websocket("/ws/{call_id}")
async def call_websocket(
    websocket: WebSocket,
    call_id: str,
    tenant_id: str = Query(default="unknown"),
):
    """
    Twilio Media Streams WebSocket 엔드포인트.
    오디오 청크를 수신해 LangGraph 파이프라인에 투입한다.
    """
    await websocket.accept()
    logger.info(f"WebSocket 연결 수락 call_id={call_id}")

    turn_index = 0
    utterance_buffer = bytearray()
    silence_chunk_count = 0
    in_speech = False
    stall_messages: dict = {}   # "start" 이벤트에서 로드
    channel_opened = False
    empty_stt_count = 0         # 연속 빈 STT 횟수 — stt_node 에 주입, 결과에서 업데이트
    last_activity_at = time.monotonic()  # 침묵 타이머 기준 시각
    silence_alert_count = 0     # 침묵 알림 횟수 (0: 없음, 1: 확인 멘트, 2: escalation)

    # Phase 2 — 통화 전체에서 누적되는 맥락 정보 (Intent Router 입력)
    # tenant_name / is_within_hours 는 start 이벤트에서 1회 채움.
    # last_intent / last_question / last_assistant_text / clarify_count 는 매 턴 후 갱신.
    session_view: dict = {
        "tenant_name": "고객센터",
        "is_within_hours": True,
        "turn_count": 0,
        "last_intent": None,
        "last_question": None,
        "last_assistant_text": None,
        "clarify_count": 0,
    }

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                logger.info(f"call_id={call_id} Twilio Media Stream connected")

            elif event == "start":
                stream_sid = msg.get("streamSid", "")
                custom_params = msg.get("start", {}).get("customParameters", {})
                tenant_id = custom_params.get("tenant_id", tenant_id)
                logger.info(
                    f"call_id={call_id} stream_sid={stream_sid} tenant_id={tenant_id}"
                )

                # RFC 001 v0.2 — TTSOutputChannel open + stall_messages pre-load
                try:
                    await tts_channel.open(
                        call_id=call_id,
                        tenant_id=tenant_id,
                        websocket=websocket,
                        stream_sid=stream_sid,
                    )
                    channel_opened = True
                except TypeError:
                    # MockTTSOutputChannel 은 websocket/stream_sid 인자를 모름 — 기본 시그니처로 재시도
                    await tts_channel.open(call_id=call_id, tenant_id=tenant_id)
                    channel_opened = True
                stall_messages = await _session_service.get_stall_messages(tenant_id)

                # Deepgram 스트리밍 연결 개설 — 통화 전체에서 재사용
                try:
                    await _streaming_stt.open(call_id)
                except Exception as _stt_err:
                    logger.warning("STT 스트리밍 연결 실패 call_id=%s: %s (prerecorded 폴백)", call_id, _stt_err)

                # 연결 즉시 인사말 발송 — 영업시간 내/외에 따라 다른 멘트
                within_hours = await _session_service.is_within_business_hours(tenant_id)
                greeting = await get_greeting(tenant_id, within_hours)

                # Intent Router 가 사용할 통화 메타 채우기 (turn 0 부터 활용)
                tenant_name = await get_tenant_name(tenant_id)
                session_view["tenant_name"] = tenant_name
                session_view["is_within_hours"] = within_hours

                await tts_channel.push_response(
                    call_id=call_id, text=greeting, response_path="greeting"
                )
                logger.info(
                    "call_id=%s greeting 발송 within_hours=%s tenant_name=%s",
                    call_id, within_hours, tenant_name,
                )
                # greeting 재생 완료 후부터 침묵 타이머 시작 — 재생 예상 시간만큼 유예
                # emit 시점 기준이라 재생 완료 전 알림 방지 (greeting ≈ 11~13초 음성)
                last_activity_at = time.monotonic() + _GREETING_PLAY_BUFFER_SEC

            elif event == "media":
                track = msg["media"].get("track", "inbound")
                if track != "inbound":
                    continue

                mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                pcm_bytes = mulaw_to_pcm16(mulaw_bytes)
                rms = audioop.rms(pcm_bytes, 2)
                await _streaming_stt.send(call_id, pcm_bytes)

                if rms >= _SPEECH_RMS_THRESHOLD:
                    # 발화 구간 — 버퍼에 누적, 묵음 카운터 초기화
                    utterance_buffer.extend(pcm_bytes)
                    silence_chunk_count = 0
                    in_speech = True
                elif in_speech:
                    # 발화 직후 묵음 — trailing silence 포함 누적
                    utterance_buffer.extend(pcm_bytes)
                    silence_chunk_count += 1

                    trigger = (
                        silence_chunk_count >= _SILENCE_CHUNKS_TO_END
                        or len(utterance_buffer) >= _MAX_UTTERANCE_BYTES
                    )
                    if trigger:
                        if len(utterance_buffer) >= _MIN_UTTERANCE_BYTES:
                            chunk = bytes(utterance_buffer)
                            logger.info(
                                f"call_id={call_id} | 발화 완성 "
                                f"{len(chunk)} bytes ({len(chunk)/32000:.1f}s) → 그래프 실행"
                            )
                            streaming_transcript = await _streaming_stt.flush_transcript(call_id)
                            state: CallState = {
                                "call_id": call_id,
                                "tenant_id": tenant_id,
                                "turn_index": turn_index,
                                "audio_chunk": chunk,
                                "is_speech": False,
                                "is_speaker_verified": False,
                                "raw_transcript": streaming_transcript,
                                "normalized_text": "",
                                "query_embedding": [],
                                "cache_hit": False,
                                "knn_intent": None,
                                "knn_confidence": 0.0,
                                "primary_intent": None,
                                "secondary_intents": [],
                                "routing_reason": None,
                                "session_view": session_view,
                                "rag_results": [],
                                "response_text": "",
                                "response_path": "",
                                "reviewer_applied": False,
                                "reviewer_verdict": None,
                                "is_timeout": False,
                                "error": None,
                                "stall_messages": stall_messages,
                                "stall_delay_sec": 1.0,
                                "empty_stt_count": empty_stt_count,
                            }
                            result = await _graph.ainvoke(state)
                            turn_index += 1
                            # 빈 STT 카운터 + 침묵 타이머 업데이트
                            _result = result if isinstance(result, dict) else {}
                            if _result.get("raw_transcript"):
                                empty_stt_count = 0
                                silence_alert_count = 0
                            else:
                                empty_stt_count = _result.get("empty_stt_count", empty_stt_count)
                            # TTS 응답 재생 완료 후부터 침묵 타이머 시작
                            # response_text 길이로 예상 재생 시간 추정 (한국어 ≈ 5자/초)
                            _resp_len = len(_result.get("response_text", "") or "")
                            _play_buffer = _resp_len / _KO_TTS_CHARS_PER_SEC
                            last_activity_at = time.monotonic() + _play_buffer

                            # Phase 2 — 다음 턴의 Intent Router 가 사용할 맥락 누적.
                            # last_assistant_text 는 길면 잘라서 프롬프트 부담을 줄인다.
                            _new_intent = _result.get("primary_intent")
                            session_view["turn_count"] += 1
                            session_view["last_intent"] = _new_intent
                            session_view["last_question"] = (
                                _result.get("normalized_text") or streaming_transcript
                            )
                            session_view["last_assistant_text"] = (
                                (_result.get("response_text") or "")[:200]
                            )
                            # clarify 가 연속될 때만 카운터 누적, 다른 intent 가 한 번이라도 끼면 리셋
                            if _new_intent == "intent_clarify":
                                session_view["clarify_count"] += 1
                            else:
                                session_view["clarify_count"] = 0
                        else:
                            logger.debug(
                                f"call_id={call_id} | 발화 무시 "
                                f"(too short: {len(utterance_buffer)} bytes)"
                            )

                        utterance_buffer.clear()
                        silence_chunk_count = 0
                        in_speech = False

                else:
                    # in_speech=False, rms<threshold → 발화 없는 진짜 침묵 구간
                    # 5초 침묵: 확인 멘트 / 추가 10초 침묵: escalation 멘트
                    if channel_opened and silence_alert_count < 2:
                        now = time.monotonic()
                        elapsed = now - last_activity_at
                        if silence_alert_count == 1 and elapsed >= _SILENCE_SECOND_SEC:
                            logger.info("call_id=%s 침묵 escalation (2차)", call_id)
                            await tts_channel.push_response(
                                call_id=call_id,
                                text=_MSG_SILENCE_ESCALATION,
                                response_path="escalation",
                            )
                            silence_alert_count = 2
                            last_activity_at = now
                        elif silence_alert_count == 0 and elapsed >= _SILENCE_FIRST_SEC:
                            logger.info("call_id=%s 침묵 확인 멘트 (1차, %.1f초)", call_id, elapsed)
                            await tts_channel.push_response(
                                call_id=call_id,
                                text=_MSG_SILENCE_CHECK,
                                response_path="silence_check",
                            )
                            silence_alert_count = 1
                            last_activity_at = now

            elif event == "stop":
                logger.info(f"[{CALL_ENDED}] call_id={call_id}")
                reset_resample_state()
                await _streaming_stt.close(call_id)
                if channel_opened:
                    await tts_channel.flush(call_id)
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket 종료 call_id={call_id}")
        reset_resample_state()
        await _streaming_stt.close(call_id)
        if channel_opened:
            await tts_channel.flush(call_id)
    except Exception as e:
        logger.error(f"call_id={call_id} WebSocket 오류: {e}")
        reset_resample_state()
        await _streaming_stt.close(call_id)
        if channel_opened:
            await tts_channel.flush(call_id)
