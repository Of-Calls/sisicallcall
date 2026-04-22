import audioop
import base64
import json

import asyncpg

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.agents.conversational.graph import build_call_graph
from app.agents.conversational.state import CallState
from app.core.events import CALL_ENDED, CALL_STARTED
from app.services.session.redis_session import RedisSessionService
from app.services.tts.channel import tts_channel
from app.utils.audio import mulaw_to_pcm16, reset_resample_state
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 그래프 싱글톤 (앱 기동 시 1회 컴파일)
_graph = build_call_graph()
_session_service = RedisSessionService()

# 발화 감지 파라미터 — Silero VAD(주미) 구현 전 임시 에너지 기반 묵음 감지
_SPEECH_RMS_THRESHOLD = 800    # RMS 임계값 (묵음 vs 발화)
_SILENCE_CHUNKS_TO_END = 40    # 발화 종료 판단 묵음 청크 수 (~800ms, 청크당 ~20ms)
_MIN_UTTERANCE_BYTES = 16000   # 최소 발화 길이 (~500ms at 16kHz 16-bit)
_MAX_UTTERANCE_BYTES = 320000  # 최대 발화 길이 (~10s)


async def _resolve_tenant_id(to_number: str) -> str:
    """Twilio To 필드(전화번호 또는 SIP URI)로 tenant UUID 조회.
    미등록 시 raw 값을 그대로 반환하고 경고 로그 출력.
    """
    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            # 1차: raw 값 그대로 매칭
            row = await conn.fetchrow(
                "SELECT id FROM tenants WHERE twilio_number = $1", to_number
            )
            if row:
                return str(row["id"])

            # 2차: SIP URI에서 user part 추출 후 매칭
            # sip:+821012345678@host → +821012345678
            # sip:50@host           → 50
            if to_number.startswith("sip:"):
                user_part = to_number.split("sip:")[1].split("@")[0]
                row = await conn.fetchrow(
                    "SELECT id FROM tenants WHERE twilio_number = $1", user_part
                )
                if row:
                    return str(row["id"])
        finally:
            await conn.close()
    except Exception as e:
        logger.warning(f"tenant 조회 실패 to={to_number} err={e}")

    logger.warning(f"미등록 tenant to={to_number} — raw 값으로 진행 (DB에 번호 등록 필요)")
    return to_number


@router.post("/incoming")
async def incoming_call(request: Request):
    """
    Twilio가 전화 수신 시 호출하는 webhook.
    TwiML을 반환해 Twilio Media Streams WebSocket 연결을 지시한다.
    """
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    to_number = form.get("To", "unknown")
    tenant_id = await _resolve_tenant_id(to_number)

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

            elif event == "media":
                track = msg["media"].get("track", "inbound")
                if track != "inbound":
                    continue

                mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                pcm_bytes = mulaw_to_pcm16(mulaw_bytes)
                rms = audioop.rms(pcm_bytes, 2)

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
                            state: CallState = {
                                "call_id": call_id,
                                "tenant_id": tenant_id,
                                "turn_index": turn_index,
                                "audio_chunk": chunk,
                                "is_speech": False,
                                "is_speaker_verified": False,
                                "raw_transcript": "",
                                "normalized_text": "",
                                "query_embedding": [],
                                "cache_hit": False,
                                "knn_intent": None,
                                "knn_confidence": 0.0,
                                "primary_intent": None,
                                "secondary_intents": [],
                                "routing_reason": None,
                                "session_view": {},
                                "rag_results": [],
                                "response_text": "",
                                "response_path": "",
                                "reviewer_applied": False,
                                "reviewer_verdict": None,
                                "is_timeout": False,
                                "error": None,
                                "stall_messages": stall_messages,
                                "stall_delay_sec": 1.0,
                            }
                            await _graph.ainvoke(state)
                            turn_index += 1
                        else:
                            logger.debug(
                                f"call_id={call_id} | 발화 무시 "
                                f"(too short: {len(utterance_buffer)} bytes)"
                            )

                        utterance_buffer.clear()
                        silence_chunk_count = 0
                        in_speech = False

            elif event == "stop":
                logger.info(f"[{CALL_ENDED}] call_id={call_id}")
                reset_resample_state()
                if channel_opened:
                    await tts_channel.flush(call_id)
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket 종료 call_id={call_id}")
        reset_resample_state()
        if channel_opened:
            await tts_channel.flush(call_id)
    except Exception as e:
        logger.error(f"call_id={call_id} WebSocket 오류: {e}")
        reset_resample_state()
        if channel_opened:
            await tts_channel.flush(call_id)
