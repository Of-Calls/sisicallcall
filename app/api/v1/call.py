import base64
import csv
import json
import os
import time
from datetime import datetime

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.agents.conversational.graph import build_call_graph
from app.agents.conversational.state import CallState
from app.core.events import CALL_ENDED, CALL_STARTED
from app.services.speaker_verify.titanet import TitaNetSpeakerVerifyService
from app.services.vad.silero import SileroVADService
from app.utils.audio import mulaw_to_pcm16, reset_resample_state
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 싱글톤 — 앱 기동 시 1회만 모델 로드
_graph = build_call_graph()
_vad = SileroVADService()
_speaker_verify = TitaNetSpeakerVerifyService()

# call_id별 enrollment 상태 (Redis 전환 전 인메모리 관리)
_enrollment_buffers: dict[str, bytearray] = {}
_enrollment_done: dict[str, bool] = {}
_verify_results: dict[str, list[dict]] = {}

# 평가 상태
_eval_labels: dict[str, str] = {}  # call_id → "self" | "background" | "unknown"

_PCM_BYTES_PER_SEC = 16000 * 2  # 16kHz, 16-bit mono
_MODEL_LABEL = settings.titanet_model_name  # titanet_small
_EVAL_DIR = "C:/kdy/Final/eval_results"


def _get_csv_path(call_id: str) -> str:
    os.makedirs(_EVAL_DIR, exist_ok=True)
    return f"{_EVAL_DIR}/{_MODEL_LABEL}_{call_id}.csv"


def _write_csv_row(call_id: str, row: dict) -> None:
    path = _get_csv_path(call_id)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "type", "label", "similarity", "verified", "latency_ms"],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(row)


async def _try_enroll(call_id: str, pcm_chunk: bytes) -> None:
    if _enrollment_done.get(call_id, False):
        return

    if call_id not in _enrollment_buffers:
        _enrollment_buffers[call_id] = bytearray()

    _enrollment_buffers[call_id].extend(pcm_chunk)

    accumulated_bytes = len(_enrollment_buffers[call_id])
    target_bytes = int(settings.enrollment_target_sec * _PCM_BYTES_PER_SEC)
    accumulated_ms = int(accumulated_bytes / _PCM_BYTES_PER_SEC * 1000)
    target_ms = int(settings.enrollment_target_sec * 1000)
    is_ready = accumulated_bytes >= target_bytes

    logger.info(
        f"call_id={call_id} enrollment 누적={accumulated_ms}ms / 목표={target_ms}ms "
        f"ready={is_ready}"
    )

    if is_ready:
        _enrollment_done[call_id] = True
        enrollment_audio = bytes(_enrollment_buffers.pop(call_id))
        try:
            t0 = time.perf_counter()
            await _speaker_verify.extract_and_store(enrollment_audio, call_id)
            enrollment_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[EVAL] call_id={call_id} enrollment latency={enrollment_ms:.1f}ms")
            _write_csv_row(call_id, {
                "timestamp": datetime.now().isoformat(),
                "type": "enrollment",
                "label": "",
                "similarity": "",
                "verified": "",
                "latency_ms": f"{enrollment_ms:.1f}",
            })
        except Exception as e:
            logger.error(f"call_id={call_id} enrollment 처리 실패: {e}")


def _print_verify_summary(call_id: str) -> None:
    results = _verify_results.get(call_id, [])
    if not results:
        logger.info(f"[VERIFY SUMMARY] call_id={call_id} 검증 없음 (enrollment 미완료)")
        return

    total = len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / total

    self_results = [r for r in results if r["label"] == "self"]
    bg_results = [r for r in results if r["label"] == "background"]

    tar = sum(1 for r in self_results if r["verified"]) / len(self_results) * 100 if self_results else None
    frr = sum(1 for r in self_results if not r["verified"]) / len(self_results) * 100 if self_results else None
    far = sum(1 for r in bg_results if r["verified"]) / len(bg_results) * 100 if bg_results else None

    self_sim = [r["similarity"] for r in self_results if r["similarity"] > 0]
    bg_sim = [r["similarity"] for r in bg_results if r["similarity"] > 0]

    lines = [
        f"\n{'=' * 55}",
        f"[VERIFY SUMMARY] model={_MODEL_LABEL} call_id={call_id}",
        f"  총 검증 횟수        : {total}",
        f"  평균 레이턴시       : {avg_latency:.1f}ms",
    ]
    if tar is not None:
        lines += [
            f"  --- 본인(self) {len(self_results)}회 ---",
            f"  TAR (통과율)        : {tar:.1f}%",
            f"  FRR (거부율)        : {frr:.1f}%",
            f"  similarity 평균     : {sum(self_sim)/len(self_sim):.4f}" if self_sim else "  similarity 없음",
        ]
    if far is not None:
        lines += [
            f"  --- 배경(background) {len(bg_results)}회 ---",
            f"  FAR (barge-in 비율) : {far:.1f}%",
            f"  similarity 평균     : {sum(bg_sim)/len(bg_sim):.4f}" if bg_sim else "  similarity 없음",
        ]
    lines.append("=" * 55)
    logger.info("\n".join(lines))


def _cleanup_enrollment(call_id: str) -> None:
    _print_verify_summary(call_id)
    _enrollment_buffers.pop(call_id, None)
    _enrollment_done.pop(call_id, None)
    _verify_results.pop(call_id, None)
    _eval_labels.pop(call_id, None)
    _speaker_verify.cleanup(call_id)


@router.post("/mark/{call_id}")
async def mark_label(call_id: str, label: str = Query(...)):
    """현재 구간 라벨 설정. label=self | background"""
    _eval_labels[call_id] = label
    logger.info(f"[EVAL] call_id={call_id} label → {label}")
    return {"call_id": call_id, "label": label}


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

                # 320ms 분량(16kHz, 16-bit mono = 10240 bytes) 누적 후 VAD → 그래프 투입
                if len(audio_buffer) >= 10240:
                    chunk = bytes(audio_buffer)
                    audio_buffer.clear()

                    vad_result = await _vad.detect(chunk)
                    logger.debug(
                        f"call_id={call_id} VAD is_speech={vad_result['is_speech']} "
                        f"score={vad_result['score']} duration_ms={vad_result['duration_ms']}"
                    )
                    if not vad_result["is_speech"]:
                        continue

                    await _try_enroll(call_id, chunk)

                    is_verified = False
                    if _enrollment_done.get(call_id, False):
                        t0 = time.perf_counter()
                        is_verified, similarity = await _speaker_verify.verify(chunk, call_id)
                        latency_ms = (time.perf_counter() - t0) * 1000

                        label = _eval_labels.get(call_id, "unknown")
                        _verify_results.setdefault(call_id, []).append({
                            "verified": is_verified,
                            "similarity": similarity,
                            "label": label,
                            "latency_ms": latency_ms,
                        })
                        _write_csv_row(call_id, {
                            "timestamp": datetime.now().isoformat(),
                            "type": "verify",
                            "label": label,
                            "similarity": f"{similarity:.4f}",
                            "verified": is_verified,
                            "latency_ms": f"{latency_ms:.1f}",
                        })
                        logger.info(
                            f"[EVAL] call_id={call_id} label={label} "
                            f"similarity={similarity:.4f} verified={is_verified} "
                            f"latency={latency_ms:.1f}ms"
                        )

                    state: CallState = {
                        "call_id": call_id,
                        "tenant_id": tenant_id,
                        "turn_index": turn_index,
                        "audio_chunk": chunk,
                        "is_speech": False,
                        "is_speaker_verified": is_verified,
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

                    await _graph.ainvoke(state)
                    turn_index += 1

            elif event == "stop":
                logger.info(f"[{CALL_ENDED}] call_id={call_id}")
                _cleanup_enrollment(call_id)
                reset_resample_state()
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket 종료 call_id={call_id}")
        _cleanup_enrollment(call_id)
        reset_resample_state()
    except Exception as e:
        logger.error(f"call_id={call_id} WebSocket 오류: {e}")
        _cleanup_enrollment(call_id)
        reset_resample_state()
