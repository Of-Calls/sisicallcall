"""
모바일 얼굴 인증 테스트 — 등록 + SMS 발송만 실행.

핸드폰 SMS 링크 클릭 후 브라우저에서 liveness·카메라 인증을 직접 진행합니다.
(test_auth_flow.py는 liveness까지 API로 완료하므로 모바일 테스트에 사용 불가)

실행:
  python scripts/mobile_auth_test.py <사진경로> <수신번호>
  예) python scripts/mobile_auth_test.py scripts/myface.jpg +821047722480
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

BASE_URL = "http://localhost:8000"
TENANT_ID = "ba2bf499-6fcc-4340-b3dd-9341f8bcc915"
CUSTOMER_REF = "test_customer_001"


async def main(image_path: str, phone: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. 얼굴 등록
        print("\n[1/2] 얼굴 등록 중...")
        with open(image_path, "rb") as f:
            r = await client.post(
                f"{BASE_URL}/auth/register",
                params={"tenant_id": TENANT_ID, "customer_ref": CUSTOMER_REF},
                files={"file": (Path(image_path).name, f, "image/jpeg")},
            )
        if r.status_code != 200:
            print(f"❌ 얼굴 등록 실패: {r.status_code} {r.json()}")
            return
        print(f"✅ 등록 완료: {r.json()}")

        # 2. 세션 생성 + SMS 발송 (여기서 멈춤 — 이후는 핸드폰이 담당)
        print("\n[2/2] 인증 세션 생성 + SMS 발송 중...")
        r = await client.post(f"{BASE_URL}/auth/verify", json={
            "tenant_id": TENANT_ID,
            "customer_ref": CUSTOMER_REF,
            "customer_phone": phone,
            "call_id": "mobile-test-001",
        })
        data = r.json()
        if r.status_code != 200:
            print(f"❌ 세션 생성 실패: {r.status_code} {data}")
            return

        auth_id = data.get("auth_id")
        print(f"✅ SMS 발송 완료")
        print(f"   auth_id: {auth_id}")
        print(f"\n📱 핸드폰 SMS 링크를 클릭해서 인증을 진행하세요.")
        print(f"   (liveness → 카메라 촬영 → 결과 확인)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python scripts/mobile_auth_test.py <사진경로> <수신번호>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1], sys.argv[2]))
