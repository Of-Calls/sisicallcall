import asyncio
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
from app.services.speaker_verify.cam_plus_plus import CAMPlusPlusSpeakerVerifyService
from app.services.speaker_verify.eres2net import ERes2NetSpeakerVerifyService
from app.services.speaker_verify.titanet import TitaNetSpeakerVerifyService
from app.services.vad.silero import SileroVADService
from app.utils.audio import mulaw_to_pcm16, reset_resample_state
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

_graph = build_call_graph()
_vad = SileroVADService()

# Step1 우승 모델 + Step2 비교군 (Step1 결과 확정 전까지 titanet_large 고정)
_SERVICES = {
    "titanet_large": TitaNetSpeakerVerifyService(),
    "cam_plus_plus": CAMPlusPlusSpeakerVerifyService(),
    "eres2net_base": ERes2NetSpeakerVerifyService(variant="base"),
    "eres2net_v2": ERes2NetSpeakerVerifyService(variant="v2"),
}

_THRESHOLDS: dict[str, float] = {
    "titanet_large": settings.titanet_similarity_threshold,
    "cam_plus_plus": settings.cam_similarity_threshold,
    "eres2net_base": settings.eres2net_base_similarity_threshold,
    "eres2net_v2": settings.eres2net_v2_similarity_threshold,
}

_PCM_BYTES_PER_SEC = 16000 * 2
_CHUNK_BYTES = 10240
_VERIFY_WINDOW_BYTES = int(1.2 * _PCM_BYTES_PER_SEC)
_EVAL_DIR = "C:/kdy/Final/eval_results"
_LABEL_TO_SCORE_TYPE = {"self": "genuine", "background": "impostor"}

# 모델별 · call별 상태
_enrollment_buffers: dict[str, dict[str, bytearray]] = {n: {} for n in _SERVICES}
_enrollment_done: dict[str, dict[str, bool]] = {n: {} for n in _SERVICES}
_verify_buffers: dict[str, dict[str, bytearray]] = {n: {} for n in _SERVICES}
_verify_results: dict[str, dict[str, list]] = {n: {} for n in _SERVICES}

# call별 (모델 공유)
_eval_labels: dict[str, str] = {}
_call_start_times: dict[str, float] = {}
_enrollment_latencies: dict[str, dict[str, float]] = {n: {} for n in _SERVICES}


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _get_csv_path(model_name: str, call_id: str) -> str:
    os.makedirs(_EVAL_DIR, exist_ok=True)
    return f"{_EVAL_DIR}/{model_name}_{call_id}.csv"


def _rename_csv_on_end(model_name: str, call_id: str) -> None:
    old_path = _get_csv_path(model_name, call_id)
    if not os.path.exists(old_path):
        return
    end_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.rename(old_path, f"{_EVAL_DIR}/{model_name}_{end_time}.csv")


def _write_csv_row(model_name: str, call_id: str, row: dict) -> None:
    path = _get_csv_path(model_name, call_id)
    is_new = not os.path.exists(path)
    start = _call_start_times.get(call_id, time.perf_counter())
    row["elapsed_sec"] = f"{time.perf_counter() - start:.1f}"
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "elapsed_sec", "type", "label", "score_type", "similarity", "verified", "latency_ms"],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ── Enrollment ────────────────────────────────────────────────────────────────

async def _try_enroll(model_name: str, call_id: str, pcm_chunk: bytes) -> None:
    if _enrollment_done[model_name].get(call_id, False):
        return

    _enrollment_buffers[model_name].setdefault(call_id, bytearray()).extend(pcm_chunk)

    accumulated_bytes = len(_enrollment_buffers[model_name][call_id])
    target_bytes = int(settings.enrollment_target_sec * _PCM_BYTES_PER_SEC)

    if accumulated_bytes < target_bytes:
        return

    _enrollment_done[model_name][call_id] = True
    enrollment_audio = bytes(_enrollment_buffers[model_name].pop(call_id))
    try:
        t0 = time.perf_counter()
        await _SERVICES[model_name].extract_and_store(enrollment_audio, call_id)
        enrollment_ms = (time.perf_counter() - t0) * 1000
        _enrollment_latencies[model_name][call_id] = enrollment_ms
        logger.info(f"[EVAL] {model_name} call_id={call_id} enrollment latency={enrollment_ms:.1f}ms")
        _write_csv_row(model_name, call_id, {
            "timestamp": datetime.now().isoformat(),
            "type": "enrollment",
            "label": "",
            "score_type": "",
            "similarity": "",
            "verified": "",
            "latency_ms": f"{enrollment_ms:.1f}",
        })
    except Exception as e:
        logger.error(f"{model_name} call_id={call_id} enrollment 실패: {e}")
        _enrollment_done[model_name][call_id] = False


# ── Verify ────────────────────────────────────────────────────────────────────

async def _try_verify(model_name: str, call_id: str, verify_audio: bytes) -> tuple[bool, float]:
    try:
        t0 = time.perf_counter()
        is_verified, similarity = await _SERVICES[model_name].verify(verify_audio, call_id)
        latency_ms = (time.perf_counter() - t0) * 1000

        label = _eval_labels.get(call_id, "unknown")
        score_type = _LABEL_TO_SCORE_TYPE.get(label, "unknown")

        _verify_results[model_name].setdefault(call_id, []).append({
            "verified": is_verified,
            "similarity": similarity,
            "label": label,
            "latency_ms": latency_ms,
        })
        _write_csv_row(model_name, call_id, {
            "timestamp": datetime.now().isoformat(),
            "type": "verify",
            "label": label,
            "score_type": score_type,
            "similarity": f"{similarity:.4f}",
            "verified": is_verified,
            "latency_ms": f"{latency_ms:.1f}",
        })
        logger.info(
            f"[EVAL] {model_name} call_id={call_id} label={label} "
            f"similarity={similarity:.4f} verified={is_verified} latency={latency_ms:.1f}ms"
        )
        return is_verified, similarity
    except Exception as e:
        logger.error(f"{model_name} call_id={call_id} verify 실패: {e}")
        return False, 0.0


# ── Summary / Cleanup ─────────────────────────────────────────────────────────

def _print_verify_summary(model_name: str, call_id: str) -> None:
    results = _verify_results[model_name].get(call_id, [])
    if not results:
        logger.info(f"[VERIFY SUMMARY] {model_name} call_id={call_id} 검증 없음")
        return

    total = len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / total
    enrollment_ms = _enrollment_latencies[model_name].get(call_id)
    genuine = [r for r in results if r["label"] == "self"]
    impostor = [r for r in results if r["label"] == "background"]

    tar = sum(1 for r in genuine if r["verified"]) / len(genuine) * 100 if genuine else None
    frr = sum(1 for r in genuine if not r["verified"]) / len(genuine) * 100 if genuine else None
    far = sum(1 for r in impostor if r["verified"]) / len(impostor) * 100 if impostor else None
    rej_rate = (100 - far) if far is not None else None

    genuine_sim = [r["similarity"] for r in genuine if r["similarity"] > 0]
    impostor_sim = [r["similarity"] for r in impostor if r["similarity"] > 0]

    lines = [
        f"\n{'=' * 55}",
        f"[VERIFY SUMMARY] model={model_name} call_id={call_id}",
        f"  총 검증 횟수        : {total}",
        f"  평균 레이턴시       : {avg_latency:.1f}ms",
        f"  threshold           : {_THRESHOLDS[model_name]}",
    ]
    if enrollment_ms is not None:
        lines.append(f"  enrollment 생성시간 : {enrollment_ms:.1f}ms")
    if tar is not None:
        lines += [
            f"  --- genuine(self) {len(genuine)}회 ---",
            f"  TAR                 : {tar:.1f}%",
            f"  FRR                 : {frr:.1f}%",
            f"  similarity 평균     : {sum(genuine_sim)/len(genuine_sim):.4f}" if genuine_sim else "  similarity 없음",
            f"  similarity 최저     : {min(genuine_sim):.4f}" if genuine_sim else "  similarity 최저 없음",
        ]
    if far is not None:
        lines += [
            f"  --- impostor(background) {len(impostor)}회 ---",
            f"  FAR (barge-in율)    : {far:.1f}%",
            f"  타인 거부율         : {rej_rate:.1f}%",
            f"  similarity 평균     : {sum(impostor_sim)/len(impostor_sim):.4f}" if impostor_sim else "  similarity 없음",
        ]
    lines.append("=" * 55)
    logger.info("\n".join(lines))


def _cleanup_all(call_id: str) -> None:
    for mn in _SERVICES:
        _print_verify_summary(mn, call_id)
        _rename_csv_on_end(mn, call_id)
        _enrollment_buffers[mn].pop(call_id, None)
        _enrollment_done[mn].pop(call_id, None)
        _verify_buffers[mn].pop(call_id, None)
        _verify_results[mn].pop(call_id, None)
        _SERVICES[mn].cleanup(call_id)
    _eval_labels.pop(call_id, None)
    _call_start_times.pop(call_id, None)
    for mn in _SERVICES:
        _enrollment_latencies[mn].pop(call_id, None)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/mark/{call_id}")
async def mark_label(call_id: str, label: str = Query(...)):
    """현재 구간 라벨 설정. label=self | background"""
    _eval_labels[call_id] = label
    logger.info(f"[EVAL] call_id={call_id} label → {label}")
    return {"call_id": call_id, "label": label}


@router.post("/incoming")
async def incoming_call(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    tenant_id = form.get("To", "unknown")

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
    await websocket.accept()
    _call_start_times[call_id] = time.perf_counter()
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
                logger.info(f"call_id={call_id} stream_sid={stream_sid} tenant_id={tenant_id}")

            elif event == "media":
                track = msg["media"].get("track", "inbound")
                if track != "inbound":
                    continue

                mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                pcm_bytes = mulaw_to_pcm16(mulaw_bytes)
                audio_buffer.extend(pcm_bytes)

                if len(audio_buffer) < _CHUNK_BYTES:
                    continue

                chunk = bytes(audio_buffer)
                audio_buffer.clear()

                vad_result = await _vad.detect(chunk)
                logger.debug(
                    f"call_id={call_id} VAD is_speech={vad_result['is_speech']} "
                    f"score={vad_result['score']} duration_ms={vad_result['duration_ms']}"
                )
                if not vad_result["is_speech"]:
                    continue

                # 4개 모델 enrollment 병렬 실행
                await asyncio.gather(
                    *[_try_enroll(mn, call_id, chunk) for mn in _SERVICES],
                    return_exceptions=True,
                )

                # 각 모델별 verify window 누적
                for mn in _SERVICES:
                    if not _enrollment_done[mn].get(call_id, False):
                        continue
                    _verify_buffers[mn].setdefault(call_id, bytearray()).extend(chunk)

                # window가 찬 모델만 골라 병렬 verify
                verify_tasks = {}
                for mn in _SERVICES:
                    buf = _verify_buffers[mn].get(call_id)
                    if buf and len(buf) >= _VERIFY_WINDOW_BYTES:
                        verify_audio = bytes(buf[-_VERIFY_WINDOW_BYTES:])
                        del buf[:_CHUNK_BYTES]
                        verify_tasks[mn] = _try_verify(mn, call_id, verify_audio)

                verify_outcomes: dict[str, tuple[bool, float]] = {}
                if verify_tasks:
                    results = await asyncio.gather(*verify_tasks.values(), return_exceptions=True)
                    for mn, res in zip(verify_tasks.keys(), results):
                        if not isinstance(res, Exception):
                            verify_outcomes[mn] = res

                # 상태 머신용 is_verified — titanet_large 결과 우선
                is_verified = verify_outcomes.get("titanet_large", (False, 0.0))[0]

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
                _cleanup_all(call_id)
                reset_resample_state()
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket 종료 call_id={call_id}")
        _cleanup_all(call_id)
        reset_resample_state()
    except Exception as e:
        logger.error(f"call_id={call_id} WebSocket 오류: {e}")
        _cleanup_all(call_id)
        reset_resample_state()
