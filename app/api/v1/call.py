import asyncio
import audioop
import base64
import json
import time
from typing import Optional

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
from app.services.speaker_verify.titanet import get_titanet_service
from app.services.stt.deepgram_streaming import DeepgramStreamingSTTService
from app.services.tts.channel import tts_channel
from app.services.vad.webrtc_vad import WebRTCVADService
from app.utils.audio import mulaw_to_pcm16, reset_resample_state
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 그래프 싱글톤 (앱 기동 시 1회 컴파일)
_graph = build_call_graph()
_session_service = RedisSessionService()
_streaming_stt = DeepgramStreamingSTTService()
# barge-in verify 게이트용 — VAD 1차, TitaNet 2차. 둘 다 모듈 싱글톤.
_bargein_vad = WebRTCVADService()

# 발화 감지 파라미터 — Silero VAD(주미) 구현 전 임시 에너지 기반 묵음 감지
_SPEECH_RMS_THRESHOLD = 1200   # 일반 발화 임계값 (TTS 에코 필터링 포함)
# Barge-in 감지 임계값 — TTS 재생 중 사용자 발화로 인한 echo 와 진짜 인터럽트를 구분.
# 일반 임계값(1200) 보다 높여야 echo 가 false positive 로 잡히지 않음. 실통화 측정 후 조정.
_BARGEIN_RMS_THRESHOLD = 2400
# Barge-in 후 silence_check 가 즉시 끼어드는 것 방지. 사용자가 인터럽트 후 새 말 꺼낼
# 자연스러운 텀. 너무 길면 (>5초) 진짜 무응답 escalation 이 늦어짐.
_BARGEIN_GRACE_SEC = 3.0
_SILENCE_CHUNKS_TO_END = 40    # 발화 종료 판단 묵음 청크 수 (~800ms, 청크당 ~20ms)
_MIN_UTTERANCE_BYTES = 16000   # 최소 발화 길이 (~500ms at 16kHz 16-bit)
_MAX_UTTERANCE_BYTES = 320000  # 최대 발화 길이 (~10s)

# 침묵 감지 파라미터
_SILENCE_FIRST_SEC = 10.0      # 첫 번째 침묵 알림 기준 (초)
_SILENCE_SECOND_SEC = 10.0     # 첫 알림 후 추가 대기 → escalation (초)
# TTS 재생 예상 속도 — Azure Korean TTS(YuJin 기본 속도) 실측 ≈ 3자/초
# 실통화 로그 기준: 60자 → 19.6초 오디오 = 3.06자/초
# ainvoke 완료 후 response_text 길이로 재생 시간 추정, 침묵 타이머 유예에 사용
_KO_TTS_CHARS_PER_SEC = 3.0
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

    Barge-in 지원:
    - 그래프 실행(_run_turn)은 백그라운드 task 로 띄워 메인 루프가 항상 inbound 처리.
    - TTS 송신 중 사용자 발화(rms ≥ _BARGEIN_RMS_THRESHOLD) 감지 시
      tts_channel.cancel() + turn_task.cancel() + 새 발화로 전환.
    - 끊긴 응답 원문은 interrupted_response_text 로 다음 turn state 에 주입,
      intent_router_llm 이 컨텍스트로 활용.
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
    session_view: dict = {
        "tenant_name": "고객센터",
        "is_within_hours": True,
        "turn_count": 0,
        "last_intent": None,
        "last_question": None,
        "last_assistant_text": None,
        "clarify_count": 0,
        "rag_miss_count": 0,  # FAQ RAG miss 누적 — faq_branch_node 가 분기에 사용
    }

    # Barge-in 상태 — 메인 루프 + _run_turn 공유
    turn_task: Optional[asyncio.Task] = None
    interrupted_response_text: str = ""
    # 한 발화당 verify 1회 가드 — utterance_buffer 0.8초 도달 시 verify, 발화 종료 시 reset
    bargein_verify_attempted: bool = False

    async def _run_turn(state: CallState, streaming_transcript: str) -> None:
        """그래프 실행 + 결과 후처리. 백그라운드 task 로 메인 루프와 분리.

        nonlocal 변수(empty_stt_count, silence_alert_count, last_activity_at)와
        session_view dict 를 직접 갱신. asyncio 단일 이벤트 루프이므로 race 없음.
        barge-in 시 cancel 되면 갱신 자체가 일어나지 않아 다음 turn 의
        session_view 가 안전하게 유지된다.
        """
        nonlocal empty_stt_count, silence_alert_count, last_activity_at
        try:
            result = await _graph.ainvoke(state)
        except asyncio.CancelledError:
            logger.info("call_id=%s turn cancelled (barge-in)", call_id)
            raise
        except Exception as e:
            logger.error("call_id=%s turn 실행 오류: %s", call_id, e)
            return

        _result = result if isinstance(result, dict) else {}
        if _result.get("raw_transcript"):
            empty_stt_count = 0
            silence_alert_count = 0
        else:
            empty_stt_count = _result.get("empty_stt_count", empty_stt_count)
        # TTS 응답 재생 완료 후부터 침묵 타이머 시작 — 길이로 재생 시간 추정
        _resp_text = _result.get("response_text") or ""
        _play_buffer = len(_resp_text) / _KO_TTS_CHARS_PER_SEC
        last_activity_at = time.monotonic() + _play_buffer

        # 다음 턴의 Intent Router 가 사용할 맥락 누적
        _new_intent = _result.get("primary_intent")
        session_view["turn_count"] += 1
        session_view["last_intent"] = _new_intent
        session_view["last_question"] = (
            _result.get("normalized_text") or streaming_transcript
        )
        if _resp_text:
            session_view["last_assistant_text"] = _resp_text[:200]
        if _new_intent == "intent_clarify":
            session_view["clarify_count"] += 1
        else:
            session_view["clarify_count"] = 0

        # FAQ RAG miss 누적 — response_path=="faq" 인 turn 만 누적, 다른 path 는 reset
        if _result.get("response_path") == "faq":
            session_view["rag_miss_count"] = _result.get("rag_miss_count", 0)
        else:
            session_view["rag_miss_count"] = 0

    async def _cancel_turn_task() -> None:
        """진행 중 turn task 가 있으면 cancel + 정리."""
        nonlocal turn_task
        if turn_task is not None and not turn_task.done():
            turn_task.cancel()
            try:
                await turn_task
            except (asyncio.CancelledError, Exception):
                pass
        turn_task = None

    async def _attempt_bargein_verify() -> None:
        """utterance_buffer 가 0.8초 채웠을 때 VAD + TitaNet 으로 BARGE-IN 검증.

        통과 시: 끊긴 응답 보존, TTS cancel, turn cancel, grace 적용. utterance_buffer 와
        STT 누적은 보존 (현재 발화가 다음 turn 의 입력이 됨).
        미통과 시: 무시. 일반 발화 처리는 그대로 진행 (echo / 타인이면 STT 결과로도 자연 차단).
        한 발화당 1회 시도 — bargein_verify_attempted 플래그로 가드.
        """
        nonlocal interrupted_response_text, last_activity_at, silence_alert_count
        nonlocal bargein_verify_attempted
        if not settings.bargein_verify_enabled:
            bargein_verify_attempted = True
            return

        bargein_verify_attempted = True
        chunk = bytes(utterance_buffer[: settings.bargein_verify_chunk_bytes])

        # VAD 1차 — 음성 청크 여부 (잡음/정적 차단, GPU 호출 절약)
        try:
            is_speech = await _bargein_vad.detect(chunk)
        except Exception as exc:
            logger.warning("call_id=%s bargein VAD 실패: %s — skip", call_id, exc)
            return
        if not is_speech:
            logger.debug("call_id=%s bargein verify skip — VAD non-speech", call_id)
            return

        # TitaNet 2차 — 등록 화자 vs echo / 타인. 미등록 시 bypass(True, 1.0)
        try:
            is_verified, similarity = await get_titanet_service().verify(chunk, call_id)
        except Exception as exc:
            logger.error("call_id=%s bargein verify 실패: %s — skip", call_id, exc)
            return

        channel_speaking = channel_opened and tts_channel.is_speaking(call_id)
        turn_running = turn_task is not None and not turn_task.done()
        logger.info(
            "call_id=%s BARGE-IN verify sim=%.3f verified=%s speaking=%s turn_running=%s",
            call_id, similarity, is_verified, channel_speaking, turn_running,
        )
        if not is_verified:
            return

        # ── 통과 → BARGE-IN trigger ─────────────────────────
        logger.info("call_id=%s BARGE-IN 감지 (verified)", call_id)
        if channel_opened:
            interrupted_response_text = tts_channel.current_text(call_id) or ""
            await tts_channel.cancel(call_id)
        await _cancel_turn_task()
        last_activity_at = time.monotonic() + _BARGEIN_GRACE_SEC
        silence_alert_count = 0
        # utterance_buffer / STT 누적은 보존 — 발화 종료 (800ms 묵음) 시 정상 turn invoke 가
        # 현재 발화 PCM + STT transcript 로 다음 응답 생성

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
                # tenant 가용 RAG 카테고리 1회 fetch — faq_branch_node 가 rag_miss>=2 시 안내 사용
                session_view["rag_categories"] = await _session_service.get_rag_categories(tenant_id)

                # Greeting 송신은 백그라운드 task — 메인 루프가 즉시 inbound 처리 시작 (greeting barge-in 가능)
                asyncio.create_task(
                    tts_channel.push_response(
                        call_id=call_id, text=greeting, response_path="greeting"
                    )
                )
                logger.info(
                    "call_id=%s greeting 발송 within_hours=%s tenant_name=%s",
                    call_id, within_hours, tenant_name,
                )
                # greeting 재생 완료 후부터 침묵 타이머 시작
                last_activity_at = time.monotonic() + _GREETING_PLAY_BUFFER_SEC

            elif event == "media":
                track = msg["media"].get("track", "inbound")
                if track != "inbound":
                    continue

                mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                pcm_bytes = mulaw_to_pcm16(mulaw_bytes)
                rms = audioop.rms(pcm_bytes, 2)

                # ── 일반 발화 처리 ────────────────────────────
                # Phase B: BARGE-IN 검증은 utterance_buffer 가 0.8초 채운 시점에
                # _attempt_bargein_verify() 가 VAD + TitaNet 게이트로 판단 (이 분기 끝).
                # 즉시 cancel 분기는 제거 — echo / 타인 목소리에 false-positive 차단.
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
                            if interrupted_response_text:
                                state["is_bargein"] = True
                                state["interrupted_response_text"] = interrupted_response_text
                                interrupted_response_text = ""
                            # FAQ RAG miss 누적 카운터 주입 (faq_branch_node 가 LLM 분기에 사용)
                            state["rag_miss_count"] = session_view.get("rag_miss_count", 0)
                            # tenant 가용 카테고리 주입 — Redis 에서 미리 fetch (start 이벤트)
                            state["available_categories"] = session_view.get("rag_categories", [])

                            # 이전 turn task 가 아직 살아있으면 정리 (정상 흐름에선 거의 없음)
                            await _cancel_turn_task()
                            # 백그라운드 실행 — 송신 동안에도 메인 루프가 inbound 처리
                            turn_task = asyncio.create_task(
                                _run_turn(state, streaming_transcript)
                            )
                            turn_index += 1
                        else:
                            logger.debug(
                                f"call_id={call_id} | 발화 무시 "
                                f"(too short: {len(utterance_buffer)} bytes)"
                            )

                        utterance_buffer.clear()
                        silence_chunk_count = 0
                        in_speech = False
                        bargein_verify_attempted = False  # 다음 발화부터 verify 1회 다시 가능

                else:
                    # in_speech=False, rms<threshold → 발화 없는 진짜 침묵 구간
                    # 5초 침묵: 확인 멘트 / 추가 10초 침묵: escalation 멘트
                    # push_response 는 백그라운드 task — 메인 루프 블록 방지
                    #
                    # 응답 생성·송신 중 가드: ainvoke 백그라운드 task 진행 중이거나
                    # TTS 송신 중이면 침묵 알림 skip + last_activity_at 계속 reset.
                    # 사용자는 응답 기다리는 중인데 "통화 중이십니까" 끼어들면 어색.
                    turn_running_now = turn_task is not None and not turn_task.done()
                    channel_speaking_now = channel_opened and tts_channel.is_speaking(call_id)
                    if turn_running_now or channel_speaking_now:
                        last_activity_at = time.monotonic()
                    elif channel_opened and silence_alert_count < 2:
                        now = time.monotonic()
                        elapsed = now - last_activity_at
                        if silence_alert_count == 1 and elapsed >= _SILENCE_SECOND_SEC:
                            logger.info("call_id=%s 침묵 escalation (2차)", call_id)
                            asyncio.create_task(
                                tts_channel.push_response(
                                    call_id=call_id,
                                    text=_MSG_SILENCE_ESCALATION,
                                    response_path="escalation",
                                )
                            )
                            silence_alert_count = 2
                            last_activity_at = now
                        elif silence_alert_count == 0 and elapsed >= _SILENCE_FIRST_SEC:
                            logger.info("call_id=%s 침묵 확인 멘트 (1차, %.1f초)", call_id, elapsed)
                            asyncio.create_task(
                                tts_channel.push_response(
                                    call_id=call_id,
                                    text=_MSG_SILENCE_CHECK,
                                    response_path="silence_check",
                                )
                            )
                            silence_alert_count = 1
                            last_activity_at = now

                # ── BARGE-IN verify trigger (per utterance, 1회) ──
                # TTS 재생 중(또는 turn 진행 중) + utterance_buffer 가 0.8초 채워졌을 때
                # _attempt_bargein_verify() 가 VAD + TitaNet 게이트로 cancel 여부 결정.
                # 결과(통과/미통과) 와 무관하게 발화 종료까지 utterance_buffer 는 누적 계속.
                channel_speaking = channel_opened and tts_channel.is_speaking(call_id)
                turn_running = turn_task is not None and not turn_task.done()
                if (
                    (channel_speaking or turn_running)
                    and not bargein_verify_attempted
                    and len(utterance_buffer) >= settings.bargein_verify_chunk_bytes
                ):
                    await _attempt_bargein_verify()

            elif event == "stop":
                logger.info(f"[{CALL_ENDED}] call_id={call_id}")
                await _cancel_turn_task()
                reset_resample_state()
                await _streaming_stt.close(call_id)
                if channel_opened:
                    await tts_channel.flush(call_id)
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket 종료 call_id={call_id}")
        await _cancel_turn_task()
        reset_resample_state()
        await _streaming_stt.close(call_id)
        if channel_opened:
            await tts_channel.flush(call_id)
    except Exception as e:
        logger.error(f"call_id={call_id} WebSocket 오류: {e}")
        await _cancel_turn_task()
        reset_resample_state()
        await _streaming_stt.close(call_id)
        if channel_opened:
            await tts_channel.flush(call_id)
