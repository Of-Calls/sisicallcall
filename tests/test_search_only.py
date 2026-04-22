"""
ChromaDB 검색 단독 테스트 — 청킹 없이 기존 저장 데이터로 검색만 검증
실행: python tests/test_search_only.py

사전 조건:
  - docker compose up -d 실행 중
  - tests/test_chunking.py로 ChromaDB에 문서 저장 완료
  - .env에 DATABASE_URL 설정
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()


async def main():
    from app.utils.config import settings
    from app.services.rag.chroma import ChromaRAGService

    # 임베더 선택 (BGE-M3 우선, 실패 시 Mock)
    try:
        from app.services.embedding.local import BGEM3LocalEmbeddingService
        embedder = BGEM3LocalEmbeddingService()
        print("임베더: BGE-M3 로컬 모델")
    except Exception as e:
        from app.services.embedding.mock import MockEmbeddingService
        embedder = MockEmbeddingService()
        print(f"임베더: Mock (BGE-M3 로드 실패 — {e})")

    # tenant_id 조회
    import asyncpg
    conn = await asyncpg.connect(settings.database_url)
    try:
        rows = await conn.fetch("SELECT id, twilio_number FROM tenants LIMIT 10")
    finally:
        await conn.close()

    if not rows:
        print("tenant 없음 — seed_postgres.py 먼저 실행하세요")
        return

    print("\n등록된 tenant:")
    for i, row in enumerate(rows):
        print(f"  [{i}] {row['twilio_number']} → {row['id']}")

    idx = input("\n검색할 tenant 번호 선택 (Enter = 0번): ").strip()
    try:
        tenant_id = str(rows[int(idx) if idx else 0]["id"])
    except (ValueError, IndexError):
        print("  잘못된 입력 — 0번 선택")
        tenant_id = str(rows[0]["id"])
    print(f"tenant_id: {tenant_id}\n")

    rag = ChromaRAGService()

    # 검색 루프
    while True:
        query = input("검색어 입력 (종료: q): ").strip()
        if query.lower() in ("q", ""):
            break

        embedding = await embedder.embed(query)
        results = await rag.search(embedding, tenant_id, top_k=3)

        if not results:
            print("  결과 없음\n")
            continue

        print(f"\n  검색 결과 ({len(results)}개):")
        for i, chunk in enumerate(results):
            print(f"\n  [{i+1}] {chunk[:200]}{'...' if len(chunk) > 200 else ''}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
