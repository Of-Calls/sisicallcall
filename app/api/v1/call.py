import base64
import json
import shutil
import wave
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.agents.conversational.graph import build_call_graph
from app.agents.conversational.nodes.stt_node.stt_node import stt_node
from app.core.events import CALL_ENDED, CALL_STARTED
from app.utils.audio import mulaw_to_pcm16, reset_resample_state
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
_RECORDINGS_DIR = Path("recordings")
_ORIGIN_RECORDINGS_DIR = _RECORDINGS_DIR / "origin"
_call_graph = build_call_graph()
VAD_MODEL_NAME = "webrtc_vad"
# 실시간 파이프라인 기본 단위: 20ms PCM16 frame
SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
FRAME_MS = 20
FRAME_BYTES = int(SAMPLE_RATE * SAMPLE_WIDTH_BYTES * (FRAME_MS / 1000))
# interim은 짧은 슬라이딩 윈도우로 자주 갱신하고,
# final은 무음 + commit delay가 충족될 때만 확정한다.
STT_INTERIM_WINDOW_MS = 400
STT_FINAL_SILENCE_MS = 600
STT_FINAL_COMMIT_DELAY_MS = 300
STT_INTERIM_MIN_BYTES = int(
    SAMPLE_RATE * SAMPLE_WIDTH_BYTES * (STT_INTERIM_WINDOW_MS / 1000)
)
STT_FINAL_TRIGGER_FRAMES = int(STT_FINAL_SILENCE_MS / FRAME_MS)
STT_FINAL_COMMIT_DELAY_FRAMES = int(STT_FINAL_COMMIT_DELAY_MS / FRAME_MS)
STT_FINAL_FRAMES = STT_FINAL_TRIGGER_FRAMES + STT_FINAL_COMMIT_DELAY_FRAMES


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
    오디오 청크를 수신해 VAD/STT 모델들을 실행한다.
    """
    await websocket.accept()
    logger.info(f"WebSocket 연결 수락 call_id={call_id}")

    audio_buffer = bytearray()
    raw_audio_16k = bytearray()
    vad_total_frames = 0
    vad_speech_frames = 0
    per_model_audio_16k: dict[str, bytearray] = {}
    stt_transcript_by_model: dict[str, list[str]] = {}
    # interim: 짧은 주기 STT용 / speech: 최종 확정 STT용 누적 버퍼
    stt_interim_buffer_by_model: dict[str, bytearray] = {}
    stt_speech_buffer_by_model: dict[str, bytearray] = {}
    # 모델별 연속 무음 frame 수(최종 확정 시점 계산에 사용)
    stt_silence_frames_by_model: dict[str, int] = {}
    finalized = False
    turn_index = 0
    scenario_name = _sanitize_path_name(tenant_id)
    recording_ext = "wav"

    async def _transcribe(model_name: str, audio_bytes: bytes) -> str:
        # STT는 노드 경유로 실행(whisper service 재사용)
        _ = model_name
        result = await stt_node({"audio_chunk": audio_bytes})
        transcript = result.get("raw_transcript", "")
        return transcript.strip() if transcript else ""

    async def _run_interim_stt(model_name: str) -> None:
        # 짧은 주기(예: 400ms)마다 임시 전사를 찍어 지연을 줄인다.
        interim_buf = stt_interim_buffer_by_model.setdefault(model_name, bytearray())
        if len(interim_buf) < STT_INTERIM_MIN_BYTES:
            return
        transcript = await _transcribe(model_name, bytes(interim_buf))
        if transcript:
            logger.info(
                "STT interim call_id=%s model=%s text=%s",
                call_id,
                model_name,
                transcript,
            )
        interim_buf.clear()

    async def _run_final_stt(model_name: str, *, force: bool = False) -> None:
        # 최종 전사는 "충분한 무음 + commit delay" 이후에만 확정(final commit)한다.
        speech_buf = stt_speech_buffer_by_model.setdefault(model_name, bytearray())
        if not speech_buf:
            return
        if (not force) and stt_silence_frames_by_model.get(
            model_name, 0
        ) < STT_FINAL_FRAMES:
            return
        transcript = await _transcribe(model_name, bytes(speech_buf))
        if transcript:
            stt_transcript_by_model.setdefault(model_name, []).append(transcript)
            logger.info(
                "STT final call_id=%s model=%s 글자수=%s",
                call_id,
                model_name,
                len(transcript),
            )
        else:
            logger.info("STT final 결과 없음 call_id=%s model=%s", call_id, model_name)
        speech_buf.clear()
        stt_interim_buffer_by_model.setdefault(model_name, bytearray()).clear()
        stt_silence_frames_by_model[model_name] = 0

    async def _flush_final_stt(force: bool = False) -> None:
        model_names = set(stt_speech_buffer_by_model.keys()) | set(
            stt_interim_buffer_by_model.keys()
        )
        for model_name in model_names:
            await _run_final_stt(model_name, force=force)

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
        if vad_total_frames > 0:
            summary = {
                VAD_MODEL_NAME: {
                    "처리프레임수": vad_total_frames,
                    "is_speech_true횟수": vad_speech_frames,
                    "is_speech_true비율": round(vad_speech_frames / vad_total_frames, 4),
                }
            }
            logger.info("통화 종료 VAD 요약 call_id=%s 상세=%s", call_id, summary)
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

                # 20ms 프레임 단위로 VAD/STT를 지속 처리
                while len(audio_buffer) >= FRAME_BYTES:
                    chunk = bytes(audio_buffer[:FRAME_BYTES])
                    del audio_buffer[:FRAME_BYTES]
                    turn_index += 1
                    vad_total_frames += 1

                    graph_state = await _call_graph.ainvoke(
                        {
                            "call_id": call_id,
                            "tenant_id": tenant_id,
                            "turn_index": turn_index,
                            "audio_chunk": chunk,
                        }
                    )
                    if not graph_state.get("is_speech", False):
                        stt_silence_frames_by_model[VAD_MODEL_NAME] = (
                            stt_silence_frames_by_model.get(VAD_MODEL_NAME, 0) + 1
                        )
                        await _run_final_stt(VAD_MODEL_NAME, force=False)
                        continue

                    vad_speech_frames += 1
                    stt_silence_frames_by_model[VAD_MODEL_NAME] = 0
                    per_model_audio_16k.setdefault(VAD_MODEL_NAME, bytearray()).extend(chunk)
                    stt_interim_buffer_by_model.setdefault(VAD_MODEL_NAME, bytearray()).extend(
                        chunk
                    )
                    stt_speech_buffer_by_model.setdefault(VAD_MODEL_NAME, bytearray()).extend(
                        chunk
                    )
                    await _run_interim_stt(VAD_MODEL_NAME)

            elif event == "stop":
                logger.info(f"[{CALL_ENDED}] call_id={call_id}")
                await _flush_final_stt(force=True)
                _finalize_call()
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket 종료 call_id={call_id}")
        await _flush_final_stt(force=True)
        _finalize_call()
    except Exception as e:
        logger.error(f"call_id={call_id} WebSocket 오류: {e}")
        await _flush_final_stt(force=True)
        _finalize_call()
