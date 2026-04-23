import base64
import json
import shutil
import wave
from datetime import datetime
from pathlib import Path
from statistics import median

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.core.events import CALL_ENDED, CALL_STARTED
from app.services.vad.benchmark import compare_vad_models
from app.services.stt.deepgram import DeepgramSTTService
from app.utils.audio import mulaw_to_pcm16, reset_resample_state
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
_RECORDINGS_DIR = Path("recordings")
_ORIGIN_RECORDINGS_DIR = _RECORDINGS_DIR / "origin"
_stt_service: DeepgramSTTService | None = None
_stt_unavailable_reason = ""
STT_MIN_BUFFER_BYTES = 16000 * 2  # 약 1초(16kHz, 16-bit mono)


def _get_stt_service() -> DeepgramSTTService | None:
    global _stt_service, _stt_unavailable_reason
    if _stt_service:
        return _stt_service
    if _stt_unavailable_reason:
        return None
    if not settings.deepgram_api_key.strip():
        _stt_unavailable_reason = "DEEPGRAM_API_KEY 없음"
        logger.warning("STT 비활성화: %s", _stt_unavailable_reason)
        return None
    try:
        _stt_service = DeepgramSTTService()
    except Exception as e:
        _stt_unavailable_reason = str(e)
        logger.warning("STT 비활성화: %s", e)
        return None
    return _stt_service


def _sanitize_path_name(name: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip()
    )
    return cleaned or "unknown"


def _save_pcm16_audio(
    file_path: Path,
    pcm16_audio: bytes,
    sample_rate: int = 16000,
    extension: str = "wav",
) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if extension == "wav":
        with wave.open(str(file_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16_audio)
        return
    if extension == "pcm":
        file_path.write_bytes(pcm16_audio)
        return
    raise ValueError(f"지원하지 않는 확장자: {extension}")


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
    오디오 청크를 수신해 VAD 모델들을 직접 실행한다.
    """
    await websocket.accept()
    logger.info(f"WebSocket 연결 수락 call_id={call_id}")

    audio_buffer = bytearray()
    raw_audio_16k = bytearray()
    vad_latency_ms_by_model: dict[str, list[float]] = {}
    vad_is_speech_by_model: dict[str, bool] = {}
    vad_true_count_by_model: dict[str, int] = {}
    per_model_audio_16k: dict[str, bytearray] = {}
    stt_transcript_by_model: dict[str, list[str]] = {}
    stt_audio_buffer_by_model: dict[str, bytearray] = {}
    finalized = False
    scenario_name = _sanitize_path_name(tenant_id)
    recording_ext = "wav"

    async def _transcribe_and_store(model_name: str, audio_bytes: bytes) -> None:
        stt_service = _get_stt_service()
        if not stt_service:
            return
        transcript = await stt_service.transcribe(audio_bytes)
        if transcript:
            stt_transcript_by_model.setdefault(model_name, []).append(transcript)
            logger.info(
                "STT 전사 성공 call_id=%s model=%s 글자수=%s",
                call_id,
                model_name,
                len(transcript),
            )
            return
        logger.info("STT 전사 결과 없음 call_id=%s model=%s", call_id, model_name)

    async def _flush_stt_buffers(force: bool = False) -> None:
        for model_name, model_buf in stt_audio_buffer_by_model.items():
            if not model_buf:
                continue
            if (not force) and len(model_buf) < STT_MIN_BUFFER_BYTES:
                continue
            await _transcribe_and_store(model_name, bytes(model_buf))
            model_buf.clear()

    def _finalize_call() -> None:
        nonlocal finalized, scenario_name, recording_ext
        if finalized:
            return
        finalized = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        scenario_dir = _RECORDINGS_DIR / scenario_name
        raw_scenario_dir = _ORIGIN_RECORDINGS_DIR / scenario_name

        # 이전 저장 파일이 남아 있으면 시나리오 디렉토리만 비운 뒤 저장
        _clear_directory_if_not_empty(scenario_dir)
        _clear_directory_if_not_empty(raw_scenario_dir)

        raw_path = raw_scenario_dir / f"raw_{call_id}_{timestamp}.{recording_ext}"
        _save_pcm16_audio(raw_path, bytes(raw_audio_16k), extension=recording_ext)
        for model_name, model_audio in per_model_audio_16k.items():
            safe_model_name = _sanitize_path_name(model_name)
            model_path = scenario_dir / f"{safe_model_name}.{recording_ext}"
            _save_pcm16_audio(model_path, bytes(model_audio), extension=recording_ext)
            transcript_path = scenario_dir / f"{safe_model_name}.txt"
            transcript_lines = stt_transcript_by_model.get(model_name, [])
            transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
        logger.info(
            "통화 오디오 저장 완료 call_id=%s scenario=%s ext=%s raw=%s 모델별파일수=%s",
            call_id,
            scenario_name,
            recording_ext,
            raw_path,
            len(per_model_audio_16k),
        )
        if vad_latency_ms_by_model:
            summary = {}
            for model, values in vad_latency_ms_by_model.items():
                if not values:
                    continue
                true_count = vad_true_count_by_model.get(model, 0)
                summary[model] = {
                    "처리된청크개수": len(values),
                    "is_speech_true횟수": true_count,
                    "is_speech_true비율": round(true_count / len(values), 4),
                    "평균ms": round(sum(values) / len(values), 3),
                    "중앙값ms": round(median(values), 3),
                }
            logger.info(
                "통화 종료 VAD 모델별 지연 요약 call_id=%s 상세=%s", call_id, summary
            )
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
                scenario_name = _sanitize_path_name(
                    custom_params.get(
                        "scenario_name", custom_params.get("scenario", tenant_id)
                    )
                )
                requested_ext = (
                    str(custom_params.get("recording_ext", "wav")).strip().lower()
                )
                recording_ext = (
                    requested_ext if requested_ext in {"wav", "pcm"} else "wav"
                )
                logger.info(
                    "call_id=%s stream_sid=%s tenant_id=%s scenario=%s ext=%s",
                    call_id,
                    stream_sid,
                    tenant_id,
                    scenario_name,
                    recording_ext,
                )

            elif event == "media":
                track = msg["media"].get("track", "inbound")
                if track != "inbound":
                    continue

                mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                pcm_bytes = mulaw_to_pcm16(mulaw_bytes)
                audio_buffer.extend(pcm_bytes)
                raw_audio_16k.extend(pcm_bytes)

                # 320ms 분량(16kHz, 16-bit mono = 10240 bytes) 누적 시 VAD 실행
                if len(audio_buffer) >= 10240:
                    chunk = bytes(audio_buffer)
                    audio_buffer.clear()

                    results = await compare_vad_models(
                        chunk,
                        excluded_models={"deepgram_vad"},
                    )
                    speech_model_names: list[str] = []
                    for result in results:
                        vad_latency_ms_by_model.setdefault(result.model, []).append(
                            result.latency_ms
                        )
                        vad_is_speech_by_model[result.model] = result.is_speech
                        if not result.is_speech:
                            continue
                        speech_model_names.append(result.model)
                        vad_true_count_by_model[result.model] = (
                            vad_true_count_by_model.get(result.model, 0) + 1
                        )
                        per_model_audio_16k.setdefault(
                            result.model, bytearray()
                        ).extend(chunk)
                    if speech_model_names:
                        for model_name in speech_model_names:
                            stt_audio_buffer_by_model.setdefault(
                                model_name, bytearray()
                            ).extend(chunk)
                        await _flush_stt_buffers(force=False)

            elif event == "stop":
                logger.info(f"[{CALL_ENDED}] call_id={call_id}")
                await _flush_stt_buffers(force=True)
                _finalize_call()
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket 종료 call_id={call_id}")
        await _flush_stt_buffers(force=True)
        _finalize_call()
    except Exception as e:
        logger.error(f"call_id={call_id} WebSocket 오류: {e}")
        await _flush_stt_buffers(force=True)
        _finalize_call()
