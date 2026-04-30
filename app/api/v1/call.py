import asyncio
import audioop
import base64
import json
import time
from typing import Optional

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.agents.conversational.graph import build_call_graph
from app.agents.post_call.trigger import run_completed_post_call_background
from app.agents.conversational.state import CallState
from app.api.v1._tenant_helpers import (
    get_greeting,
    get_tenant_name,
    resolve_tenant_id,
)
from app.core.events import CALL_ENDED, CALL_STARTED
from app.repositories.call_repo import finalize_call, insert_call
from app.repositories.transcript_repo import insert_transcript
from app.services.session.redis_session import RedisSessionService
from app.services.speaker_verify import enrollment as voiceprint_enrollment
from app.services.speaker_verify.titanet import get_titanet_service
from app.services.stt.deepgram import DeepgramSTTService
from app.services.stt.deepgram_streaming import DeepgramStreamingSTTService
from app.services.stt.keyterm_cache import get_tenant_keyterms
from app.services.tts.channel import tts_channel
from app.services.vad.silero_vad import SileroVADService
from app.utils.audio import mulaw_to_pcm16, reset_resample_state
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 그래프 싱글톤 (앱 기동 시 1회 컴파일) — 텍스트 워크플로우 전용 (2026-04-30 개편).
# audio 처리 (VAD / 화자검증 / STT / enrollment) 는 graph 진입 전 본 파일에서 처리.
_graph = build_call_graph()
_session_service = RedisSessionService()
_streaming_stt = DeepgramStreamingSTTService()
_prerecorded_stt = DeepgramSTTService()  # streaming flush 가 비어있을 때 fallback
# Silero VAD — main path 게이트 + barge-in verify 1차. 둘 다 stateless 라 같은 인스턴스
# 공유 가능. (이전: WebRTCVADService 였으나 짧은 발화 + 긴 trailing silence reject 문제로
# 2026-04-30 교체. detect(bytes)→bool 인터페이스 동일, per-call cleanup 불필요.)
_silero_vad = SileroVADService()
_bargein_vad = _silero_vad  # alias — barge-in 코드 가독성용

# 발화 감지 파라미터 — Silero VAD(주미) 구현 전 임시 에너지 기반 묵음 감지
_SPEECH_RMS_THRESHOLD = 1200   # 일반 발화 임계값 (TTS 에코 필터링 포함)
# Barge-in 감지 임계값 — TTS 재생 중 사용자 발화로 인한 echo 와 진짜 인터럽트를 구분.
# 일반 임계값(1200) 보다 높여야 echo 가 false positive 로 잡히지 않음. 실통화 측정 후 조정.
_BARGEIN_RMS_THRESHOLD = 2400
# Barge-in 후 silence_check 가 즉시 끼어드는 것 방지. 사용자가 인터럽트 후 새 말 꺼낼
# 자연스러운 텀. 너무 길면 (>5초) 진짜 무응답 escalation 이 늦어짐.
_BARGEIN_GRACE_SEC = 3.0
_SILENCE_CHUNKS_TO_END = 65    # 발화 종료 판단 묵음 청크 수 (~1300ms, 청크당 ~20ms).
                                # Deepgram UtteranceEnd 이벤트가 먼저 오면 즉시 trigger,
                                # 본 카운팅은 fallback (UtteranceEnd 미수신 시 안전망).
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

# stall ("잠시만요") 발화 트리거 — graph 진입 후 이 시간 안에 응답 emit 안 되면 발화.
# faq_branch LLM Phase2 (8초) 가 아니라 graph 진입 직후로 이동 (2026-04-28 개정).
_STALL_DELAY_SEC = 1.5
_STALL_DEFAULT_TEXT = "잠시만요, 확인해 드리겠습니다."


async def _schedule_stall_task(
    call_id: str,
    stall_messages: dict,
    delay: float = _STALL_DELAY_SEC,
) -> None:
    """graph 진입 후 delay 초 동안 응답 없으면 stall 발화. _run_turn finally 에서 cancel."""
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        return
    text = (stall_messages or {}).get("general") or _STALL_DEFAULT_TEXT
    try:
        await tts_channel.push_stall(
            call_id=call_id,
            text=text,
            audio_field="general",
        )
    except asyncio.CancelledError:
        return
    except Exception as exc:  # noqa: BLE001 — best-effort, stall 실패가 통화 흐름 차단 안 함
        logger.warning("stall scheduler push 실패 call_id=%s: %s", call_id, exc)


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
    pcm_buffer = bytearray()
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
        "auth_pending": False,  # 인증 확인 질문 발화 후 응답 대기 중
    }

    # Barge-in 상태 — 메인 루프 + _run_turn 공유
    turn_task: Optional[asyncio.Task] = None
    interrupted_response_text: str = ""
    # 한 발화당 verify 1회 가드 — pcm_buffer 0.8초 도달 시 verify, 발화 종료 시 reset
    bargein_verify_attempted: bool = False

    # PostgreSQL 영구 기록 — calls.id (UUID, RETURNING 으로 채움). insert 실패 시 None 유지.
    # transcripts INSERT 는 db_call_id 가 채워졌을 때만 시도.
    db_call_id: Optional[str] = None
    call_started_at_monotonic: Optional[float] = None

    async def _run_turn(state: CallState, streaming_transcript: str) -> None:
        """그래프 실행 + 결과 후처리. 백그라운드 task 로 메인 루프와 분리.

        nonlocal 변수(empty_stt_count, silence_alert_count, last_activity_at)와
        session_view dict 를 직접 갱신. asyncio 단일 이벤트 루프이므로 race 없음.
        barge-in 시 cancel 되면 갱신 자체가 일어나지 않아 다음 turn 의
        session_view 가 안전하게 유지된다.

        Stall scheduler: graph 진입 후 _STALL_DELAY_SEC 안에 응답 없으면 "잠시만요"
        발화. ainvoke 종료(정상/cancel/예외) 시 finally 에서 cancel.
        """
        nonlocal empty_stt_count, silence_alert_count, last_activity_at
        stall_task: Optional[asyncio.Task] = None
        if channel_opened:
            # state["stall_delay_sec"] 우선, 부재 시 모듈 상수 fallback. (Phase D)
            stall_delay = state.get("stall_delay_sec", _STALL_DELAY_SEC)
            stall_task = asyncio.create_task(
                _schedule_stall_task(call_id, stall_messages, delay=stall_delay)
            )
        try:
            try:
                result = await _graph.ainvoke(state)
            except asyncio.CancelledError:
                logger.info("call_id=%s turn cancelled (barge-in)", call_id)
                raise
            except Exception as e:
                logger.error("call_id=%s turn 실행 오류: %s", call_id, e)
                return
        finally:
            if stall_task is not None and not stall_task.done():
                stall_task.cancel()

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

        # PostgreSQL transcripts 영구 기록 — best-effort, 통화 흐름 차단 안 함.
        # barge-in cancel 시 ainvoke 가 CancelledError 로 빠져 이 블록 자체가 실행 안 됨.
        if db_call_id:
            customer_text = _result.get("normalized_text") or streaming_transcript
            if customer_text:
                await insert_transcript(
                    db_call_id=db_call_id,
                    turn_index=state["turn_index"],
                    speaker="customer",
                    text=customer_text,
                    is_barge_in=bool(state.get("is_bargein")),
                )
            if _resp_text:
                await insert_transcript(
                    db_call_id=db_call_id,
                    turn_index=state["turn_index"],
                    speaker="agent",
                    text=_resp_text,
                    response_path=_result.get("response_path"),
                    reviewer_applied=bool(_result.get("reviewer_applied")),
                    reviewer_verdict=_result.get("reviewer_verdict"),
                    is_barge_in=False,
                )

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

        # 인증 대기 상태 — auth_branch_node 가 반환한 값으로 갱신 (턴 간 유지)
        session_view["auth_pending"] = _result.get("auth_pending", False)

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
        """pcm_buffer 가 0.8초 채웠을 때 VAD + TitaNet 으로 BARGE-IN 검증.

        통과 시: 끊긴 응답 보존, TTS cancel, turn cancel, grace 적용. pcm_buffer 와
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
        chunk = bytes(pcm_buffer[: settings.bargein_verify_chunk_bytes])

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

        # TTS 재생 중(speaking=True) echo로 인한 유사도 급락 보정.
        # enrollment 완료(sim<1.0) + speaking=True → 완화 임계값 적용.
        # 숨소리/잡음은 voiceprint 패턴 없어 sim≈0.05로 여전히 차단됨.
        _ECHO_BARGEIN_THRESHOLD = 0.20
        enrollment_done = similarity < 1.0  # bypass=1.0(미등록), 실측값=등록 완료
        if channel_speaking and enrollment_done:
            effective_verified = similarity >= _ECHO_BARGEIN_THRESHOLD
        else:
            effective_verified = is_verified  # 기존 임계값 0.45

        logger.info(
            "call_id=%s BARGE-IN verify sim=%.3f verified=%s speaking=%s turn_running=%s",
            call_id, similarity, effective_verified, channel_speaking, turn_running,
        )
        if not effective_verified:
            return

        # ── 통과 → BARGE-IN trigger ─────────────────────────
        logger.info("call_id=%s BARGE-IN 감지 (verified)", call_id)
        if channel_opened:
            interrupted_response_text = tts_channel.current_text(call_id) or ""
            await tts_channel.cancel(call_id)
        await _cancel_turn_task()
        last_activity_at = time.monotonic() + _BARGEIN_GRACE_SEC
        silence_alert_count = 0
        # pcm_buffer / STT 누적은 보존 — 발화 종료 (800ms 묵음) 시 정상 turn invoke 가
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

                # tenant 메타 + STT keyterm 병렬 fetch — 통화 초기화 레이턴시 최소화
                (
                    tenant_keyterms,
                    rag_categories,
                    within_hours,
                    tenant_name,
                ) = await asyncio.gather(
                    get_tenant_keyterms(tenant_id),
                    _session_service.get_rag_categories(tenant_id),
                    _session_service.is_within_business_hours(tenant_id),
                    get_tenant_name(tenant_id),
                )
                session_view["rag_categories"] = rag_categories
                session_view["is_within_hours"] = within_hours
                session_view["tenant_name"] = tenant_name
                # Option γ — query_refine_node 가 STT 띄어쓰기 정정 시 사용
                session_view["tenant_keyterms"] = tenant_keyterms
                greeting = await get_greeting(tenant_id, within_hours)

                # Deepgram 스트리밍 연결 개설 — tenant keyterm 부스팅 주입
                try:
                    await _streaming_stt.open(call_id, keyterms=tenant_keyterms)
                except Exception as _stt_err:
                    logger.warning("STT 스트리밍 연결 실패 call_id=%s: %s (prerecorded 폴백)", call_id, _stt_err)

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

                # PostgreSQL calls 영구 기록 — best-effort. 실패 시 db_call_id 는 None
                # 으로 남고 transcripts INSERT 는 자연스럽게 skip 됨. 통화 흐름은 그대로.
                db_call_id = await insert_call(
                    tenant_id=tenant_id,
                    twilio_call_sid=call_id,
                    caller_number=None,
                )
                call_started_at_monotonic = time.monotonic()

            elif event == "media":
                track = msg["media"].get("track", "inbound")
                if track != "inbound":
                    continue

                mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                pcm_bytes = mulaw_to_pcm16(mulaw_bytes)
                rms = audioop.rms(pcm_bytes, 2)

                # ── 일반 발화 처리 ────────────────────────────
                # Phase B: BARGE-IN 검증은 pcm_buffer 가 0.8초 채운 시점에
                # _attempt_bargein_verify() 가 VAD + TitaNet 게이트로 판단 (이 분기 끝).
                # 즉시 cancel 분기는 제거 — echo / 타인 목소리에 false-positive 차단.
                await _streaming_stt.send(call_id, pcm_bytes)

                if rms >= _SPEECH_RMS_THRESHOLD:
                    # 발화 구간 — 버퍼에 누적, 묵음 카운터 초기화
                    pcm_buffer.extend(pcm_bytes)
                    silence_chunk_count = 0
                    in_speech = True
                elif in_speech:
                    # 발화 직후 묵음 — trailing silence 포함 누적
                    pcm_buffer.extend(pcm_bytes)
                    silence_chunk_count += 1

                    # Deepgram 의 의미 단위 발화 종료 신호 (utterance_end_ms 기반).
                    # silence chunk 카운팅보다 먼저 도달하면 즉시 trigger — 한국어 머뭇거림
                    # 침묵에서 조기 절단 방지 (예: "주말에 ... [생각] ... 영업하나요?").
                    utterance_end_signal = _streaming_stt.consume_utterance_end(call_id)
                    trigger = (
                        silence_chunk_count >= _SILENCE_CHUNKS_TO_END
                        or len(pcm_buffer) >= _MAX_UTTERANCE_BYTES
                        or utterance_end_signal
                    )
                    if trigger:
                        if len(pcm_buffer) >= _MIN_UTTERANCE_BYTES:
                            chunk = bytes(pcm_buffer)
                            trigger_reason = (
                                "utterance_end" if utterance_end_signal
                                else ("max_bytes" if len(pcm_buffer) >= _MAX_UTTERANCE_BYTES
                                      else "silence")
                            )
                            logger.info(
                                f"call_id={call_id} | 발화 완성 "
                                f"{len(chunk)} bytes ({len(chunk)/32000:.1f}s) "
                                f"trigger={trigger_reason} → 오디오 처리"
                            )

                            # ── 오디오 처리 (graph 진입 전, 2026-04-30 구조 개편) ──
                            # 이전: graph 안의 vad / speaker_verify / stt / enrollment 노드.
                            # 이후: 본 파일이 직접 처리. graph 는 텍스트 워크플로우만 담당.

                            # 1) STT — streaming 결과 회수, 비어있으면 prerecorded fallback
                            stt_t0 = time.monotonic()
                            transcript = await _streaming_stt.flush_transcript(call_id)
                            if not transcript:
                                try:
                                    transcript = await _prerecorded_stt.transcribe(chunk)
                                    if transcript:
                                        logger.info(
                                            "call_id=%s | STT prerecorded fallback '%s'",
                                            call_id, transcript,
                                        )
                                except Exception as exc:
                                    logger.warning(
                                        "call_id=%s | STT prerecorded 실패: %s", call_id, exc,
                                    )
                            transcript_norm = " ".join(transcript.split()) if transcript else ""
                            logger.info(
                                "[pre-graph:stt] elapsed=%.0fms call_id=%s len=%d",
                                (time.monotonic() - stt_t0) * 1000, call_id, len(transcript_norm),
                            )

                            if not transcript_norm:
                                # 빈 STT — graph 진입 skip, 횟수 누적 (escalation 후속 작업)
                                empty_stt_count += 1
                                logger.debug(
                                    "call_id=%s | 빈 STT %d회 → graph skip",
                                    call_id, empty_stt_count,
                                )
                            else:
                                empty_stt_count = 0

                                # 2) 화자 검증 — STT 결과 있으면 graph 진입 자체는 막지 않음
                                #    (route_after_speaker_verify 의 STT fallback 가드 동일 정책).
                                #    텔레메트리 + 향후 echo 차단 정책 강화 여지.
                                verify_t0 = time.monotonic()
                                try:
                                    await get_titanet_service().verify(chunk, call_id)
                                except Exception as exc:
                                    logger.error("call_id=%s | 화자 검증 실패: %s", call_id, exc)
                                logger.info(
                                    "[pre-graph:verify] elapsed=%.0fms call_id=%s",
                                    (time.monotonic() - verify_t0) * 1000, call_id,
                                )

                                # 3) Enrollment — STT 성공 발화만 누적해 voiceprint 등록.
                                #    임계 (settings.titanet_enrollment_sec) 도달 시 등록.
                                enroll_t0 = time.monotonic()
                                await voiceprint_enrollment.accumulate(call_id, chunk, transcript_norm)
                                logger.info(
                                    "[pre-graph:enroll] elapsed=%.0fms call_id=%s",
                                    (time.monotonic() - enroll_t0) * 1000, call_id,
                                )

                                # 4) Graph state — 텍스트 워크플로우 전용 (audio_chunk / is_speech /
                                #    is_speaker_verified 필드 제거됨, 2026-04-30 개편)
                                state: CallState = {
                                    "call_id": call_id,
                                    "tenant_id": tenant_id,
                                    "turn_index": turn_index,
                                    "raw_transcript": transcript,
                                    "normalized_text": transcript_norm,
                                    "query_embedding": [],
                                    "cache_hit": False,
                                    "primary_intent": None,
                                    "session_view": session_view,
                                    "rag_results": [],
                                    "response_text": "",
                                    "response_path": "",
                                    "reviewer_applied": False,
                                    "reviewer_verdict": None,
                                    "is_timeout": False,
                                    "stall_messages": stall_messages,
                                    "stall_delay_sec": _STALL_DELAY_SEC,
                                    "empty_stt_count": 0,
                                }
                                if interrupted_response_text:
                                    state["is_bargein"] = True
                                    state["interrupted_response_text"] = interrupted_response_text
                                    interrupted_response_text = ""
                                # FAQ RAG miss 누적 (faq_branch_node 가 LLM 분기에 사용)
                                state["rag_miss_count"] = session_view.get("rag_miss_count", 0)
                                # tenant 가용 카테고리 (start 이벤트에서 미리 fetch)
                                state["available_categories"] = session_view.get("rag_categories", [])
                                # 인증 대기 상태 (auth_branch_node Turn 2 판단)
                                state["auth_pending"] = session_view.get("auth_pending", False)

                                # barge-in 단순 모델 (architect 감사) — turn cancel 은 오직
                                # `_attempt_bargein_verify` (TitaNet 통과 시) 만 책임진다.
                                # 새 발화 완성 자체로는 이전 turn 을 cancel 하지 않음 — verify
                                # 실패한 noise (호흡·바람소리 등) 가 응답을 끊는 것을 차단.
                                # 백그라운드 실행 — 송신 동안에도 메인 루프가 inbound 처리
                                turn_task = asyncio.create_task(
                                    _run_turn(state, transcript_norm)
                                )
                                turn_index += 1
                        else:
                            logger.debug(
                                f"call_id={call_id} | 발화 무시 "
                                f"(too short: {len(pcm_buffer)} bytes)"
                            )

                        pcm_buffer.clear()
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
                # TTS 재생 중(또는 turn 진행 중) + pcm_buffer 가 0.8초 채워졌을 때
                # _attempt_bargein_verify() 가 VAD + TitaNet 게이트로 cancel 여부 결정.
                # 결과(통과/미통과) 와 무관하게 발화 종료까지 pcm_buffer 는 누적 계속.
                channel_speaking = channel_opened and tts_channel.is_speaking(call_id)
                turn_running = turn_task is not None and not turn_task.done()
                if (
                    (channel_speaking or turn_running)
                    and not bargein_verify_attempted
                    and len(pcm_buffer) >= settings.bargein_verify_chunk_bytes
                ):
                    await _attempt_bargein_verify()

            elif event == "stop":
                logger.info(f"[{CALL_ENDED}] call_id={call_id}")
                await _cancel_turn_task()
                reset_resample_state()
                await _streaming_stt.close(call_id)
                if channel_opened:
                    await tts_channel.flush(call_id)
                if db_call_id:
                    _duration = (
                        int(time.monotonic() - call_started_at_monotonic)
                        if call_started_at_monotonic else None
                    )
                    await finalize_call(db_call_id, status="completed", duration_sec=_duration)
                    asyncio.create_task(
                        run_completed_post_call_background(
                            call_id=call_id,
                            tenant_id=tenant_id,
                        )
                    )
                else:
                    logger.warning(
                        "post-call auto run skip call_id=%s reason=db_call_id_missing",
                        call_id,
                    )
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket 종료 call_id={call_id}")
        await _cancel_turn_task()
        reset_resample_state()
        await _streaming_stt.close(call_id)
        if channel_opened:
            await tts_channel.flush(call_id)
        if db_call_id:
            _duration = (
                int(time.monotonic() - call_started_at_monotonic)
                if call_started_at_monotonic else None
            )
            await finalize_call(db_call_id, status="abandoned", duration_sec=_duration)
    except Exception as e:
        logger.error(f"call_id={call_id} WebSocket 오류: {e}")
        await _cancel_turn_task()
        reset_resample_state()
        await _streaming_stt.close(call_id)
        if channel_opened:
            await tts_channel.flush(call_id)
        if db_call_id:
            _duration = (
                int(time.monotonic() - call_started_at_monotonic)
                if call_started_at_monotonic else None
            )
            await finalize_call(db_call_id, status="error", duration_sec=_duration)
    finally:
        # per-call 메모리 해제 — enrollment 모듈 전역 dict + titanet voiceprint.
        # 종료 경로(stop / WebSocketDisconnect / Exception) 무엇이든 항상 정리.
        try:
            voiceprint_enrollment.cleanup(call_id)
            get_titanet_service().cleanup(call_id)
        except Exception as e:  # noqa: BLE001 — cleanup 실패가 통화 종료 흐름 차단 안 함
            logger.warning("call_id=%s 종료 cleanup 실패: %s", call_id, e)
