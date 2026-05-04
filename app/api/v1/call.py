import base64
import csv
import html
import json

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response

from app.pipeline.runner import TripleStreamContext, log_call_stt_latency_summary
from app.services.speaker_verify import enrollment as voice_enrollment
from app.services.speaker_verify.onnx_pipeline import get_finetuned_onnx_service
from app.services.speaker_verify.titanet_compare import cleanup_compare_for_call
from app.services.speaker_verify.titanet_compare import _resolve_compare_csv_path
from app.utils.config import settings
from app.utils.logger import get_logger

router = APIRouter()
_logger = get_logger(__name__)


@router.post("/incoming")
async def incoming_call(request: Request):
    host = request.headers.get("host", "")
    ws_url = f"wss://{host}/call/ws"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}" />
  </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@router.websocket("/ws")
async def call_ws(websocket: WebSocket):
    await websocket.accept()
    _logger.info("[WS] Twilio 연결됨")

    triple = TripleStreamContext()
    stream_sid = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                _logger.info("[WS] connected")

            elif event == "start":
                start = msg.get("start") or {}
                stream_sid = start.get("streamSid") or msg.get("streamSid")
                call_sid = start.get("callSid")
                triple.reset()
                _logger.info("[WS] start streamSid=%s callSid=%s", stream_sid, call_sid)
                try:
                    await triple.prepare_deepgram_streaming(
                        stream_sid if isinstance(stream_sid, str) else "no-stream"
                    )
                except Exception as e:
                    _logger.exception("[WS] Deepgram 스트리밍 준비 실패: %s", e)

            elif event == "media":
                mulaw = base64.b64decode(msg["media"]["payload"])
                call_id = stream_sid or "no-stream"
                await triple.feed_media_chunk(mulaw, call_id)

            elif event == "stop":
                _logger.info("[WS] stop")
                break

    except WebSocketDisconnect:
        _logger.info("[WS] 연결 끊김")
    finally:
        await triple.shutdown_deepgram_streaming()
        if stream_sid:
            log_call_stt_latency_summary(triple.stt_latency, stream_sid=stream_sid)
            get_finetuned_onnx_service().cleanup(stream_sid)
            cleanup_compare_for_call(stream_sid)
            voice_enrollment.cleanup(stream_sid)


def _require_call_debug_routes() -> None:
    if not settings.call_debug_routes_enabled:
        raise HTTPException(status_code=404, detail="Not Found")


def _read_verify_compare_rows(*, tail: int, call_id: str | None) -> tuple[str, list[dict[str, str]]]:
    path = _resolve_compare_csv_path()
    if not path.is_file():
        return str(path), []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if call_id:
        rows = [r for r in rows if r.get("call_id") == call_id]
    if tail > 0 and len(rows) > tail:
        rows = rows[-tail:]
    return str(path), rows


@router.get("/debug/twilio-webhook-hint")
async def debug_twilio_webhook_hint(request: Request) -> JSONResponse:
    """Twilio 콘솔에 넣을 URL 안내 (CALL_DEBUG_ROUTES_ENABLED=true 일 때만)."""
    _require_call_debug_routes()
    base = str(request.base_url).rstrip("/")
    host = request.headers.get("host", "")
    ws_hint = f"wss://{host}/call/ws" if host else "(Host 헤더 기준으로 Twilio Media Stream 과 동일 호스트)"
    return JSONResponse(
        {
            "twilio_phone_voice_config": "A call comes in → Webhook / HTTP POST",
            "incoming_twiml_url": f"{base}/call/incoming",
            "media_stream_websocket": ws_hint,
            "speaker_verify_compare_enabled": settings.speaker_verify_compare_enabled,
            "speaker_verify_compare_csv_path": str(_resolve_compare_csv_path()),
            "speaker_verify_compare_baseline_onnx_path": (
                (settings.speaker_verify_compare_baseline_onnx_path or "").strip() or None
            ),
            "speaker_verify_nemo_baseline_nemo_path": (
                (settings.speaker_verify_nemo_baseline_nemo_path or "").strip() or None
            ),
            "view_csv_table_in_browser": f"{base}/call/debug/verify-compare?format=html",
            "view_csv_json": f"{base}/call/debug/verify-compare?format=json&tail=100",
        }
    )


@router.get("/debug/verify-compare", response_model=None)
async def debug_verify_compare(
    tail: int = Query(50, ge=1, le=500),
    call_id: str | None = Query(None, description="streamSid 등으로 필터 후 tail 적용"),
    format: str = Query("json", description="json | html"),
) -> Response:
    """화자검증 비교 CSV tail 조회 (CALL_DEBUG_ROUTES_ENABLED=true 일 때만)."""
    _require_call_debug_routes()
    fmt = (format or "json").strip().lower()
    if fmt not in ("json", "html"):
        raise HTTPException(status_code=422, detail="format 은 json 또는 html")

    path_str, rows = _read_verify_compare_rows(tail=tail, call_id=call_id)
    if fmt == "html":
        esc = html.escape
        parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>speaker_verify_compare</title>",
            "<style>body{font-family:sans-serif;} table{border-collapse:collapse;} th,td{border:1px solid #ccc;padding:4px 8px;font-size:12px;}</style>",
            "</head><body>",
            f"<h2>{esc('speaker_verify_compare.csv')}</h2>",
            f"<p><code>{esc(path_str)}</code> — 표시 행 수: {len(rows)} (tail={tail})</p>",
        ]
        if rows:
            cols = list(rows[0].keys())
            parts.append("<table><thead><tr>")
            parts.extend(f"<th>{esc(c)}</th>" for c in cols)
            parts.append("</tr></thead><tbody>")
            for r in rows:
                parts.append("<tr>")
                parts.extend(f"<td>{esc(str(r.get(c, '')))}</td>" for c in cols)
                parts.append("</tr>")
            parts.append("</tbody></table>")
        else:
            parts.append("<p>(파일 없음 또는 행 없음)</p>")
        parts.append("</body></html>")
        return HTMLResponse("".join(parts))

    return JSONResponse(
        {
            "csv_path": path_str,
            "row_count": len(rows),
            "tail": tail,
            "call_id_filter": call_id,
            "rows": rows,
        }
    )
