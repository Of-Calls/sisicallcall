"""
얼굴 인증 전체 플로우 테스트 스크립트.

사전 조건:
  1. uvicorn app.main:app --reload 로 서버 실행
  2. .env 에 AUTH_ENABLE_TEST_REGISTER=true 설정
  3. 테스트할 얼굴 사진 파일(JPEG) 준비

실행:
  python scripts/test_auth_flow.py <사진경로> <수신번호>
  예) python scripts/test_auth_flow.py my_face.jpg +821047722480
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

BASE_URL = "http://localhost:8000"
TENANT_ID = "ba2bf499-6fcc-4340-b3dd-9341f8bcc915"  # 한밭식당
CUSTOMER_REF = "test_customer_001"


async def step(label: str, coro):
    print(f"\n{'='*50}")
    print(f"STEP: {label}")
    print('='*50)
    result = await coro
    print(f"결과: {result}")
    return result


async def register_face(client: httpx.AsyncClient, image_path: str) -> bool:
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        r = await client.post(
            f"{BASE_URL}/auth/register",
            params={"tenant_id": TENANT_ID, "customer_ref": CUSTOMER_REF},
            files=files,
        )
    data = r.json()
    print(f"  status={r.status_code} body={data}")
    return r.status_code == 200


async def create_session(client: httpx.AsyncClient, phone: str) -> str | None:
    r = await client.post(f"{BASE_URL}/auth/verify", json={
        "tenant_id": TENANT_ID,
        "customer_ref": CUSTOMER_REF,
        "customer_phone": phone,
        "call_id": "test-call-001",
    })
    data = r.json()
    print(f"  status={r.status_code} body={data}")
    return data.get("auth_id")


async def get_liveness(client: httpx.AsyncClient, auth_id: str) -> str | None:
    r = await client.get(f"{BASE_URL}/auth/{auth_id}/liveness")
    data = r.json()
    print(f"  status={r.status_code} instructions={data.get('instructions')} token={data.get('token','')[:20]}...")
    return data.get("token")


async def complete_liveness(client: httpx.AsyncClient, auth_id: str, token: str) -> bool:
    r = await client.post(f"{BASE_URL}/auth/{auth_id}/liveness", json={"token": token})
    data = r.json()
    print(f"  status={r.status_code} body={data}")
    return r.status_code == 200


async def verify_face(client: httpx.AsyncClient, auth_id: str, image_path: str) -> dict:
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        r = await client.post(f"{BASE_URL}/auth/{auth_id}/face", files=files)
    data = r.json()
    print(f"  status={r.status_code} body={data}")
    return data


async def get_status(client: httpx.AsyncClient, auth_id: str) -> dict:
    r = await client.get(f"{BASE_URL}/auth/{auth_id}/status")
    data = r.json()
    print(f"  status={r.status_code} body={data}")
    return data


async def main(image_path: str, phone: str):
    print(f"\n[시시콜콜 얼굴 인증 플로우 테스트]")
    print(f"  사진: {image_path}")
    print(f"  번호: {phone}")
    print(f"  tenant: {TENANT_ID} (한밭식당)")
    print(f"  ref: {CUSTOMER_REF}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. 얼굴 등록
        ok = await step("1. 얼굴 등록 (POST /auth/register)", register_face(client, image_path))
        if not ok:
            print("❌ 얼굴 등록 실패 — 종료")
            return

        # 2. 인증 세션 생성 + SMS 발송
        auth_id = await step("2. 세션 생성 + SMS 발송 (POST /auth/verify)", create_session(client, phone))
        if not auth_id:
            print("❌ 세션 생성 실패 — 종료")
            return
        print(f"  → auth_id: {auth_id}")

        # 3. Liveness 지시 발급
        token = await step("3. Liveness 지시 발급 (GET /auth/{id}/liveness)", get_liveness(client, auth_id))
        if not token:
            print("❌ Liveness 토큰 발급 실패 — 종료")
            return

        # 4. Liveness 완료 보고 (실제 서비스에선 프론트가 MediaPipe 수행 후 호출)
        ok = await step("4. Liveness 완료 (POST /auth/{id}/liveness)", complete_liveness(client, auth_id, token))
        if not ok:
            print("❌ Liveness 검증 실패 — 종료")
            return

        # 5. 얼굴 인증
        result = await step("5. 얼굴 인증 (POST /auth/{id}/face)", verify_face(client, auth_id, image_path))

        # 6. 최종 상태 확인
        status = await step("6. 최종 상태 (GET /auth/{id}/status)", get_status(client, auth_id))

    print(f"\n{'='*50}")
    if status.get("face_verified"):
        print("✅ 전체 인증 플로우 성공!")
    else:
        print("❌ 얼굴 인증 실패 (사진 확인 필요)")
    print('='*50)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python scripts/test_auth_flow.py <사진경로> <수신번호>")
        print("예)    python scripts/test_auth_flow.py my_face.jpg +821047722480")
        sys.exit(1)
    asyncio.run(main(sys.argv[1], sys.argv[2]))
