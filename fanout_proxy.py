"""
Fan-out proxy: Twilio → 3개 모델 서버 동시 전달
- POST /call/incoming : TwiML 생성 (ws URL은 proxy 자신으로)
- WS   /call/ws/{call_id} : Twilio 오디오를 3개 백엔드에 동시 전달

실행:
  venv/Scripts/uvicorn fanout_proxy:app --port 8000

백엔드 포트:
  8001 = TitaNet-Large  (sisicallcall)
  8002 = TitaNet-Small  (sisicallcall-TitaNet-Small)
  8003 = ECAPA-TDNN     (sisicallcall-ECAPA-TDNN)
"""

import asyncio

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

app = FastAPI()

PROXY_BASE_URL = "rylie-uninfuriated-lavinia.ngrok-free.dev"

BACKENDS = {
    "TitaNet-Small": "ws://localhost:8002",
    "ECAPA-TDNN":    "ws://localhost:8003",
    "KDT-71":        "ws://localhost:8004",
}

BACKEND_HTTP = {
    "TitaNet-Small": "http://localhost:8002",
    "ECAPA-TDNN":    "http://localhost:8003",
    "KDT-71":        "http://localhost:8004",
}


@app.post("/call/incoming")
async def incoming_call(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    tenant_id = form.get("To", "unknown")

    ws_url = f"wss://{PROXY_BASE_URL}/call/ws/{call_sid}"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}">
      <Parameter name="tenant_id" value="{tenant_id}"/>
    </Stream>
  </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@app.websocket("/call/ws/{call_id}")
async def websocket_fanout(websocket: WebSocket, call_id: str):
    await websocket.accept()

    # 3개 백엔드에 동시 연결
    backend_conns: dict[str, websockets.WebSocketClientProtocol] = {}
    for name, base in BACKENDS.items():
        url = f"{base}/call/ws/{call_id}"
        try:
            conn = await websockets.connect(url)
            backend_conns[name] = conn
            print(f"[proxy] {name} 연결 완료: {url}")
        except Exception as e:
            print(f"[proxy] {name} 연결 실패: {e}")

    if not backend_conns:
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_text()
            await asyncio.gather(
                *[conn.send(data) for conn in backend_conns.values()],
                return_exceptions=True,
            )
    except WebSocketDisconnect:
        print(f"[proxy] call_id={call_id} Twilio 연결 종료")
    finally:
        await asyncio.gather(
            *[conn.close() for conn in backend_conns.values()],
            return_exceptions=True,
        )
        print(f"[proxy] call_id={call_id} 백엔드 연결 모두 종료")


@app.post("/call/mark/{call_id}")
async def mark_label(call_id: str, label: str):
    async with httpx.AsyncClient() as client:
        await asyncio.gather(
            *[client.post(f"{base}/call/mark/{call_id}", params={"label": label})
              for base in BACKEND_HTTP.values()],
            return_exceptions=True,
        )
    return {"call_id": call_id, "label": label}
