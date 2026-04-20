"""
Redis 시드 — tenant별 운영시간 및 상담원 가용성 초기값

실행: python db/seed/seed_redis.py

설정 키:
  - tenant:{tenant_id}:business_hours  (Hash)
  - tenant:{tenant_id}:agent_availability  (Hash)

주의:
  - PostgreSQL seed 먼저 실행 필요 (tenant_id 조회 위함).
  - 멱등성: HSET으로 같은 키 덮어쓰기 허용.
"""

import asyncio
import os
import sys

import asyncpg
import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

if not DATABASE_URL:
    print("❌ DATABASE_URL 환경변수가 없습니다.", file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# tenant별 운영시간 / 상담원 가용성
# ==============================================================================
TENANT_CONFIGS = {
    "+821000000001": {  # 서울중앙병원
        "business_hours": {
            "mon": "09:00-18:00",
            "tue": "09:00-18:00",
            "wed": "09:00-18:00",
            "thu": "09:00-18:00",
            "fri": "09:00-18:00",
            "sat": "09:00-13:00",
            "sun": "closed",
        },
        "agent_availability": {
            "total": "3",
            "available": "3",
        },
    },
    "+821000000002": {  # 한밭식당
        "business_hours": {
            "mon": "11:00-22:00",
            "tue": "11:00-22:00",
            "wed": "11:00-22:00",
            "thu": "11:00-22:00",
            "fri": "11:00-22:00",
            "sat": "11:00-22:00",
            "sun": "11:00-22:00",
        },
        "agent_availability": {
            "total": "1",
            "available": "1",
        },
    },
}


async def seed():
    # PostgreSQL에서 tenant_id 조회
    pg = await asyncpg.connect(DATABASE_URL)
    try:
        tenant_rows = await pg.fetch("SELECT id, name, twilio_number FROM tenants")
    finally:
        await pg.close()

    if not tenant_rows:
        print("❌ tenants 테이블이 비어있습니다. seed_postgres.py를 먼저 실행하세요.", file=sys.stderr)
        sys.exit(1)

    # Redis 연결
    r = redis.from_url(REDIS_URL, decode_responses=True)
    try:
        print("🌱 Redis 시드 시작")

        for row in tenant_rows:
            twilio_number = row["twilio_number"]
            tenant_id_no_hyphen = str(row["id"]).replace("-", "")
            config = TENANT_CONFIGS.get(twilio_number)

            if not config:
                print(f"  ⚠️  {row['name']}: 설정 없음 — 건너뜀")
                continue

            # business_hours
            bh_key = f"tenant:{tenant_id_no_hyphen}:business_hours"
            await r.delete(bh_key)  # 기존 키 초기화 후 재설정
            await r.hset(bh_key, mapping=config["business_hours"])

            # agent_availability
            aa_key = f"tenant:{tenant_id_no_hyphen}:agent_availability"
            await r.delete(aa_key)
            await r.hset(aa_key, mapping=config["agent_availability"])

            print(f"  ✅ {row['name']}: business_hours + agent_availability 설정")

        # 최종 통계
        keys = await r.keys("tenant:*")
        print(f"\n📊 현재 tenant:* 키 개수: {len(keys)}")
        print("✅ Redis 시드 완료\n")

    finally:
        await r.aclose()


if __name__ == "__main__":
    asyncio.run(seed())
