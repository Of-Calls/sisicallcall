import audioop
import base64
import json
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.services.stt.deepgram import DeepgramSTTService
from app.services.vad.silero_vad import SileroVADService

router = APIRouter()
_stt = DeepgramSTTService()
_vad = SileroVADService()

_VAD_FRAME_BYTES = 1024  # linear16 16kHz, 512 samples
_SILENCE_THRESHOLD = 30  # 연속 침묵 VAD 프레임 수 (~960ms)


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
    print("[WS] Twilio 연결됨")

    audio_buffer = bytearray()  # STT용 (mulaw 8kHz 누적)
    pcm_buffer = bytearray()    # VAD용 (linear16 16kHz 누적)
    ratecv_state = None         # audioop.ratecv state — 8k→16k 보간 연속성 유지
    silence_count = 0
    had_speech = False          # 현재 누적 구간에 실제 발화가 있었는지

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                print("[WS] connected")

            elif event == "start":
                print("[WS] start")
                audio_buffer.clear()
                pcm_buffer.clear()
                ratecv_state = None
                silence_count = 0
                had_speech = False

            elif event == "media":
                mulaw = base64.b64decode(msg["media"]["payload"])
                audio_buffer.extend(mulaw)

                # mulaw 8kHz → linear16 8kHz → linear16 16kHz
                pcm_8k = audioop.ulaw2lin(mulaw, 2)
                pcm_16k, ratecv_state = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, ratecv_state)
                pcm_buffer.extend(pcm_16k)

                # 정확히 1024 bytes 단위로 잘라 VAD 실행 (한 청크가 여러 프레임일 수 있음)
                while len(pcm_buffer) >= _VAD_FRAME_BYTES:
                    frame = bytes(pcm_buffer[:_VAD_FRAME_BYTES])
                    del pcm_buffer[:_VAD_FRAME_BYTES]

                    is_speech = await _vad.detect(frame)

                    if is_speech:
                        silence_count = 0
                        had_speech = True
                    else:
                        silence_count += 1
                        if silence_count >= _SILENCE_THRESHOLD and had_speech:
                            print(f"[VAD] 침묵 감지 — {len(audio_buffer)} bytes → STT")
                            transcript = await _stt.transcribe(bytes(audio_buffer))
                            print(f"[STT] '{transcript}'")
                            audio_buffer.clear()
                            silence_count = 0
                            had_speech = False

            elif event == "stop":
                print("[WS] stop")
                if audio_buffer:
                    transcript = await _stt.transcribe(bytes(audio_buffer))
                    print(f"[STT] 마지막 발화: '{transcript}'")
                break

    except WebSocketDisconnect:
        print("[WS] 연결 끊김")
