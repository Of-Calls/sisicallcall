"""
SMS 발송 테스트 스크립트.

사전 조건:
  .env 에 SOLAPI_API_KEY / SOLAPI_API_SECRET / SOLAPI_SENDER_NUMBER 설정

실행:
  python scripts/test_sms.py +821012345678
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.services.sms import get_sms_service


async def main(to: str) -> None:
    svc = get_sms_service()
    body = "[시시콜콜] SMS 발송 테스트 메시지입니다."
    print(f"발송 대상: {to}")
    ok = await svc.send_sms(to=to, body=body)
    print("발송 성공 ✓" if ok else "발송 실패 ✗ — 로그 확인")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python scripts/test_sms.py +821012345678")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
