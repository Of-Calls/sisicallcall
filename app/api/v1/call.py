import base64
import json

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.pipeline.runner import TripleStreamContext, log_call_stt_latency_summary
from app.services.speaker_verify import enrollment as voice_enrollment
from app.services.speaker_verify.onnx_pipeline import get_finetuned_onnx_service
from app.services.speaker_verify.onnx_pipeline import get_onnx_pipeline_service
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
                stream_sid = msg["start"]["streamSid"]
                triple.reset()
                _logger.info("[WS] start streamSid=%s", stream_sid)

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
        if stream_sid:
            log_call_stt_latency_summary(triple.stt_latency, stream_sid=stream_sid)
            get_finetuned_onnx_service().cleanup(stream_sid)
            get_onnx_pipeline_service().cleanup(stream_sid)
            voice_enrollment.cleanup(stream_sid)
