import base64
import json
import shutil
import wave
from datetime import datetime
from pathlib import Path
from statistics import median

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.agents.conversational.graph import build_call_graph
from app.agents.conversational.state import CallState
from app.core.events import CALL_ENDED, CALL_STARTED
from app.utils.audio import mulaw_to_pcm16, reset_resample_state
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
_RECORDINGS_DIR = Path("recordings")
_ORIGIN_RECORDINGS_DIR = _RECORDINGS_DIR / "origin"

# 그래프 싱글톤 (앱 기동 시 1회 컴파일)
_graph = build_call_graph()


def _save_pcm16_wav(file_path: Path, pcm16_audio: bytes, sample_rate: int = 16000) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16_audio)


def _clear_directory_if_not_empty(dir_path: Path) -> None:
    """저장 전에 기존 파일이 있으면 디렉토리를 비운다."""
    if not dir_path.exists():
        return
    entries = list(dir_path.iterdir())
    if not entries:
        return
    for entry in entries:
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


@router.post("/incoming")
async def incoming_call(request: Request):
    """
    Twilio가 전화 수신 시 호출하는 webhook.
    TwiML을 반환해 Twilio Media Streams WebSocket 연결을 지시한다.
    """
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    tenant_id = form.get("To", "unknown")  # Twilio 수신 번호 → tenant 식별자 (임시)

    logger.info(f"[{CALL_STARTED}] call_sid={call_sid} tenant_id={tenant_id}")

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
    audio_buffer = bytearray()
    raw_audio_16k = bytearray()
    vad_pass_audio_16k = bytearray()
    vad_latency_ms_by_model: dict[str, list[float]] = {}
    vad_is_speech_by_model: dict[str, bool] = {}
    per_model_audio_16k: dict[str, bytearray] = {}
    finalized = False

    def _finalize_call() -> None:
        nonlocal finalized
        if finalized:
            return
        finalized = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 이전 저장 파일이 남아 있으면 디렉토리 정리 후 저장
        _clear_directory_if_not_empty(_RECORDINGS_DIR)
        _clear_directory_if_not_empty(_ORIGIN_RECORDINGS_DIR)

        raw_path = _ORIGIN_RECORDINGS_DIR / f"{call_id}_{timestamp}_raw.wav"
        vad_path = _RECORDINGS_DIR / f"{call_id}_{timestamp}_vad.wav"
        _save_pcm16_wav(raw_path, bytes(raw_audio_16k))
        _save_pcm16_wav(vad_path, bytes(vad_pass_audio_16k))
        for model_name, model_audio in per_model_audio_16k.items():
            model_file_name = f"{model_name}_{timestamp}.wav"
            _save_pcm16_wav(_RECORDINGS_DIR / model_file_name, bytes(model_audio))
        logger.info(
            "통화 오디오 저장 완료 call_id=%s raw=%s vad=%s 모델별파일수=%s",
            call_id,
            raw_path,
            vad_path,
            len(per_model_audio_16k),
        )
        if vad_latency_ms_by_model:
            summary = {}
            for model, values in vad_latency_ms_by_model.items():
                if not values:
                    continue
                summary[model] = {
                    "처리된청크개수": len(values),
                    "평균ms": round(sum(values) / len(values), 3),
                    "중앙값ms": round(median(values), 3),
                }
            logger.info("통화 종료 VAD 모델별 지연 요약 call_id=%s 상세=%s", call_id, summary)
        reset_resample_state()

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

            elif event == "media":
                track = msg["media"].get("track", "inbound")
                if track != "inbound":
                    continue

                mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                pcm_bytes = mulaw_to_pcm16(mulaw_bytes)
                audio_buffer.extend(pcm_bytes)
                raw_audio_16k.extend(pcm_bytes)

                # 320ms 분량(16kHz, 16-bit mono = 10240 bytes) 누적 시 그래프 투입
                if len(audio_buffer) >= 10240:
                    chunk = bytes(audio_buffer)
                    audio_buffer.clear()

                    state: CallState = {
                        "call_id": call_id,
                        "tenant_id": tenant_id,
                        "turn_index": turn_index,
                        "audio_chunk": chunk,
                        "is_speech": False,
                        "is_speaker_verified": False,
                        "vad_latency_ms_by_model": vad_latency_ms_by_model,
                        "vad_is_speech_by_model": vad_is_speech_by_model,
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
                    }

                    result_state = await _graph.ainvoke(state)
                    vad_latency_ms_by_model = result_state.get("vad_latency_ms_by_model", vad_latency_ms_by_model)
                    vad_is_speech_by_model = result_state.get("vad_is_speech_by_model", vad_is_speech_by_model)
                    for model, model_is_speech in vad_is_speech_by_model.items():
                        if not model_is_speech:
                            continue
                        per_model_audio_16k.setdefault(model, bytearray()).extend(chunk)
                    if result_state.get("is_speech", False):
                        vad_pass_audio_16k.extend(chunk)
                    turn_index += 1

            elif event == "stop":
                logger.info(f"[{CALL_ENDED}] call_id={call_id}")
                _finalize_call()
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket 종료 call_id={call_id}")
        _finalize_call()
    except Exception as e:
        logger.error(f"call_id={call_id} WebSocket 오류: {e}")
        _finalize_call()
