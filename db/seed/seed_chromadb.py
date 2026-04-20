"""
ChromaDB 시드 — tenant별 빈 컬렉션 생성

실행: python db/seed/seed_chromadb.py

컬렉션 네이밍: tenant_{tenant_id_without_hyphens}_docs
  (db_schema.md §4.1)

주의:
  - 문서는 비워둠. 실제 PDF 청킹 및 임베딩은 희영(BGE-M3) 연구 완료 후
    RAG 파이프라인에서 수행됩니다.
  - 이 시드는 컬렉션 존재성만 보장하여, 다른 팀원의 연구가 컬렉션 없음
    에러로 막히지 않도록 합니다.
"""

import os
import sys

import asyncpg
import asyncio
import chromadb
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))

if not DATABASE_URL:
    print("❌ DATABASE_URL 환경변수가 없습니다.", file=sys.stderr)
    sys.exit(1)


async def fetch_tenants():
    pg = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await pg.fetch("SELECT id, name FROM tenants")
    finally:
        await pg.close()
    return rows


def seed():
    # tenants 조회 (동기 wrapper)
    tenants = asyncio.run(fetch_tenants())

    if not tenants:
        print("❌ tenants 테이블이 비어있습니다. seed_postgres.py를 먼저 실행하세요.", file=sys.stderr)
        sys.exit(1)

    print(f"🌱 ChromaDB 시드 시작 ({CHROMA_HOST}:{CHROMA_PORT})")

    # HTTP 클라이언트 (docker-compose의 chromadb 서비스 접속)
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # heartbeat 확인
    try:
        client.heartbeat()
    except Exception as e:
        print(f"❌ ChromaDB 연결 실패: {e}", file=sys.stderr)
        print(f"   Docker 컨테이너가 실행 중인지 확인하세요: make ps", file=sys.stderr)
        sys.exit(1)

    for row in tenants:
        tenant_id_no_hyphen = str(row["id"]).replace("-", "")
        collection_name = f"tenant_{tenant_id_no_hyphen}_docs"

        # get_or_create_collection: 존재하면 그대로, 없으면 생성 (멱등)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"tenant_name": row["name"]},
        )
        count = collection.count()
        print(f"  ✅ {row['name']}: {collection_name} (문서 {count}개)")

    # 최종 통계
    all_collections = client.list_collections()
    tenant_collections = [c for c in all_collections if c.name.startswith("tenant_")]
    print(f"\n📊 현재 tenant_* 컬렉션 개수: {len(tenant_collections)}")
    print("✅ ChromaDB 시드 완료\n")


if __name__ == "__main__":
    seed()
