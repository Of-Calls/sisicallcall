"""
얼굴 인증 테스트 웹 서버 (포트 3000).

기존 코드 수정 없이 핸드폰에서 얼굴 인증 플로우를 테스트합니다.

실행 순서:
  1) 터미널 A — FastAPI 서버 시작
       uvicorn app.main:app --reload

  2) 터미널 B — 이 서버 시작
       python scripts/auth_test_server.py

  3) 터미널 C — ngrok HTTPS 터널 (카메라 권한은 HTTPS 필수)
       ngrok http 3000

  4) .env 수정 (ngrok URL 복사 후)
       AUTH_WEB_BASE_URL=https://xxxx.ngrok-free.app

  5) 인증 세션 생성 (얼굴 사진 사전 등록 필요)
       python scripts/test_auth_flow.py scripts/myface.jpg +821047722480

  6) 핸드폰 SMS 링크 클릭 → 브라우저에서 인증 진행

주의:
  - AUTH_ENABLE_TEST_REGISTER=true 가 .env 에 있어야 /auth/register 활성
  - 운영 배포 시 이 서버는 절대 사용하지 말 것
"""

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

FASTAPI_BASE = "http://localhost:8000"
PORT = 3000

# ---------------------------------------------------------------------------
# HTML 테스트 페이지 (단일 파일, MediaPipe 생략 — 토큰만 전달)
# ---------------------------------------------------------------------------
_HTML = """\
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
  <title>시시콜콜 본인인증</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:-apple-system,sans-serif;background:#f5f5f5;min-height:100vh;display:flex;align-items:center;justify-content:center}
    .card{background:#fff;border-radius:16px;padding:24px;max-width:420px;width:100%;margin:16px;box-shadow:0 2px 16px rgba(0,0,0,.08)}
    h2{font-size:20px;margin-bottom:4px}
    .subtitle{font-size:12px;color:#999;margin-bottom:20px;word-break:break-all}
    .step{border-radius:10px;padding:16px;margin:10px 0;border:1px solid #e0e0e0;transition:.2s}
    .step.active{border-color:#4CAF50;background:#f9fff9}
    .step.done{border-color:#4CAF50;background:#e8f5e9}
    .step.fail{border-color:#f44336;background:#ffebee}
    .step-title{font-weight:600;margin-bottom:8px;display:flex;align-items:center;gap:8px}
    .badge{font-size:11px;padding:2px 8px;border-radius:12px;background:#e0e0e0;color:#555}
    .badge.ok{background:#c8e6c9;color:#2e7d32}
    button{width:100%;padding:13px;font-size:15px;font-weight:600;border:none;border-radius:8px;cursor:pointer;margin-top:8px;transition:.15s}
    .btn-primary{background:#4CAF50;color:#fff}
    .btn-primary:disabled{background:#a5d6a7;cursor:not-allowed}
    .btn-blue{background:#1976D2;color:#fff}
    video,canvas{width:100%;border-radius:8px;background:#000;margin:8px 0}
    video{transform:scaleX(-1)}
    #status{text-align:center;font-size:13px;color:#666;padding:8px 0;min-height:20px}
    .instructions{font-size:14px;color:#333;margin:8px 0;padding:10px;background:#fff8e1;border-radius:6px}
    .score{font-size:28px;font-weight:700;text-align:center;margin:8px 0}
  </style>
</head>
<body>
<div class="card">
  <h2>시시콜콜 본인인증</h2>
  <p class="subtitle" id="auth-id-label"></p>

  <div class="step active" id="step1">
    <div class="step-title">1단계 <span class="badge" id="b1">대기</span></div>
    <p style="font-size:13px;color:#555;margin-bottom:8px">아래 버튼을 눌러 Liveness 확인을 시작합니다.</p>
    <div id="instructions-box" class="instructions" style="display:none"></div>
    <button class="btn-primary" id="btn-start" onclick="startLiveness()">인증 시작</button>
    <button class="btn-primary" id="btn-done-liveness" style="display:none" onclick="completeLiveness()">✅ 동작 완료</button>
  </div>

  <div class="step" id="step2" style="display:none">
    <div class="step-title">2단계 <span class="badge" id="b2">대기</span></div>
    <p style="font-size:13px;color:#555;margin-bottom:4px">정면을 바라보고 📷 버튼을 누르세요.</p>
    <video id="video" autoplay playsinline muted></video>
    <canvas id="canvas" style="display:none"></canvas>
    <button class="btn-blue" onclick="captureAndVerify()">📷 촬영 후 인증</button>
  </div>

  <div class="step" id="step3" style="display:none">
    <div class="step-title">결과 <span class="badge" id="b3"></span></div>
    <div class="score" id="score-display"></div>
    <p id="result-msg" style="text-align:center;font-size:14px;color:#555"></p>
    <button class="btn-blue" style="margin-top:12px" onclick="checkStatus()">최종 상태 확인</button>
    <pre id="status-json" style="font-size:11px;color:#777;margin-top:8px;white-space:pre-wrap"></pre>
  </div>

  <p id="status">준비 중...</p>
</div>

<script>
const authId = (() => {
  const m = location.pathname.match(/\\/auth\\/([^\\/]+)/);
  return m ? m[1] : null;
})();
document.getElementById('auth-id-label').textContent = 'session: ' + authId;
let _token = null;

function setStatus(msg) { document.getElementById('status').textContent = msg; }

async function startLiveness() {
  document.getElementById('btn-start').disabled = true;
  setStatus('Liveness 지시 요청 중...');
  try {
    const r = await fetch('/api/auth/' + authId + '/liveness');
    if (!r.ok) { const d=await r.json(); throw new Error(d.detail||r.status); }
    const data = await r.json();
    _token = data.token;
    const box = document.getElementById('instructions-box');
    box.style.display = 'block';
    box.innerHTML = '지시 동작: <b>' + data.instructions.join(' → ') + '</b><br><small style="color:#888">동작 후 완료 버튼을 누르세요 (테스트 모드: 실제 검증 생략)</small>';
    document.getElementById('btn-start').style.display = 'none';
    document.getElementById('btn-done-liveness').style.display = 'block';
    document.getElementById('b1').textContent = '진행 중';
    setStatus('지시에 따라 동작한 후 완료 버튼을 누르세요.');
  } catch(e) {
    document.getElementById('btn-start').disabled = false;
    setStatus('오류: ' + e.message);
  }
}

async function completeLiveness() {
  document.getElementById('btn-done-liveness').disabled = true;
  setStatus('Liveness 검증 중...');
  try {
    const r = await fetch('/api/auth/' + authId + '/liveness', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({token: _token}),
    });
    if (!r.ok) { const d=await r.json(); throw new Error(d.detail||r.status); }
    const data = await r.json();
    if (data.liveness_passed) {
      document.getElementById('step1').className = 'step done';
      document.getElementById('b1').textContent = '완료';
      document.getElementById('b1').className = 'badge ok';
      document.getElementById('step2').style.display = 'block';
      document.getElementById('step2').className = 'step active';
      document.getElementById('b2').textContent = '진행 중';
      await startCamera();
      setStatus('카메라가 열렸습니다. 정면을 바라보세요.');
    } else {
      setStatus('Liveness 실패. 다시 시도하세요.');
      document.getElementById('btn-done-liveness').disabled = false;
    }
  } catch(e) {
    setStatus('오류: ' + e.message);
    document.getElementById('btn-done-liveness').disabled = false;
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'user', width:640, height:480}});
    document.getElementById('video').srcObject = stream;
  } catch(e) {
    setStatus('카메라 접근 실패: ' + e.message + ' (HTTPS 필요)');
  }
}

async function captureAndVerify() {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  canvas.style.display = 'block';
  setStatus('얼굴 인증 요청 중...');

  canvas.toBlob(async (blob) => {
    const form = new FormData();
    form.append('file', blob, 'face.jpg');
    try {
      const r = await fetch('/api/auth/' + authId + '/face', {method:'POST', body:form});
      const data = await r.json();
      showResult(data, r.ok);
    } catch(e) {
      setStatus('오류: ' + e.message);
    }
  }, 'image/jpeg', 0.92);
}

function showResult(data, ok) {
  const step2 = document.getElementById('step2');
  const step3 = document.getElementById('step3');
  step2.className = 'step ' + (data.verified ? 'done' : 'fail');
  document.getElementById('b2').textContent = data.verified ? '완료' : '실패';
  document.getElementById('b2').className = 'badge ' + (data.verified ? 'ok' : '');
  step3.style.display = 'block';
  step3.className = 'step ' + (data.verified ? 'done' : 'fail');
  document.getElementById('b3').textContent = data.verified ? '인증 성공' : '인증 실패';
  document.getElementById('b3').className = 'badge ' + (data.verified ? 'ok' : '');
  document.getElementById('score-display').textContent =
    data.verified ? '✅ ' + data.similarity_score : '❌ ' + (data.similarity_score ?? '-');
  document.getElementById('result-msg').textContent = data.verified
    ? '본인 인증이 완료되었습니다.'
    : '인증 실패. 남은 시도: ' + data.attempts_remaining + '회';
  setStatus(data.verified ? '인증 완료!' : '인증 실패. 다시 촬영하거나 세션을 재시작하세요.');
}

async function checkStatus() {
  try {
    const r = await fetch('/api/auth/' + authId + '/status');
    const data = await r.json();
    document.getElementById('status-json').textContent = JSON.stringify(data, null, 2);
  } catch(e) {
    document.getElementById('status-json').textContent = '오류: ' + e.message;
  }
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP 프록시 핸들러
# ---------------------------------------------------------------------------

class AuthTestHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  [{self.address_string()}] {fmt % args}")

    # ---- GET ---------------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path

        if re.match(r"^/auth/[^/]+$", path):
            self._serve_html()
        elif path.startswith("/api/auth/"):
            self._proxy_get(path[4:])  # /api → strip
        else:
            self._send(404, b'{"error":"not found"}')

    # ---- POST --------------------------------------------------------------
    def do_POST(self):
        path = urlparse(self.path).path
        if path.startswith("/api/auth/"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            self._proxy_post(path[4:], body, self.headers.get("Content-Type", ""))
        else:
            self._send(404, b'{"error":"not found"}')

    # ---- helpers -----------------------------------------------------------
    def _serve_html(self):
        body = _HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _proxy_get(self, api_path: str):
        try:
            with httpx.Client(timeout=30.0) as c:
                r = c.get(f"{FASTAPI_BASE}{api_path}")
            self._send(r.status_code, r.content)
        except Exception as e:
            self._send(502, json.dumps({"error": str(e)}).encode())

    def _proxy_post(self, api_path: str, body: bytes, content_type: str):
        try:
            with httpx.Client(timeout=30.0) as c:
                r = c.post(
                    f"{FASTAPI_BASE}{api_path}",
                    content=body,
                    headers={"Content-Type": content_type},
                )
            self._send(r.status_code, r.content)
        except Exception as e:
            self._send(502, json.dumps({"error": str(e)}).encode())

    def _send(self, status: int, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), AuthTestHandler)
    print(f"""
┌─────────────────────────────────────────────────────┐
│        시시콜콜 인증 테스트 서버 (포트 {PORT})          │
└─────────────────────────────────────────────────────┘
  FastAPI 프록시 대상: {FASTAPI_BASE}

  [실행 순서]
  1) uvicorn app.main:app --reload
  2) python scripts/auth_test_server.py   ← 지금 실행 중
  3) ngrok http {PORT}                    ← HTTPS 필수 (카메라 권한)
  4) .env 에 AUTH_WEB_BASE_URL=https://xxxx.ngrok-free.app 설정
  5) python scripts/test_auth_flow.py scripts/myface.jpg +821047722480
  6) 핸드폰 SMS 링크 클릭

  Ctrl+C 로 종료
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료")
