"""PDF 청킹 파이프라인 통합 테스트
실행: python -m pytest tests/test_chunking.py -v -s
또는: python tests/test_chunking.py

사전 조건:
  - docker compose up -d 실행 중
  - .env에 DATABASE_URL 설정
  - tests/ 폴더에 PDF 파일 존재
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from app.services.chunking.paragraph import ParagraphChunkingService
from app.services.chunking.pdf_processor import PDFProcessor, _clean, _split_sections
from app.services.embedding.mock import MockEmbeddingService
from app.services.rag.chroma import ChromaRAGService


# ==============================================================================
# 단위 테스트: 단락 청킹
# ==============================================================================

async def test_paragraph_chunking():
    print("\n[1] 단락 청킹 테스트")
    chunker = ParagraphChunkingService()
    sample = """서울중앙병원 진료 안내입니다. 저희 병원은 내과, 외과, 정형외과, 신경과, 산부인과 등 다양한 진료과목을 운영하고 있으며 최신 의료 장비를 갖추고 있습니다.

MRI 검사는 사전 예약이 필요합니다. 검사 전 6시간 금식이 필요하며 금속 물질 제거 후 입실하셔야 합니다. 폐소공포증이 있으신 분은 사전에 담당 의사와 상담하시기 바랍니다. 검사 소요 시간은 부위에 따라 30분에서 1시간 정도입니다.

영업시간은 평일 오전 9시부터 오후 6시까지이며 토요일은 오전 9시부터 오후 1시까지 운영합니다. 일요일 및 공휴일은 휴진입니다. 야간 응급실은 24시간 운영되며 응급 환자는 언제든지 내원하실 수 있습니다."""

    chunks = await chunker.chunk(sample)
    print(f"  청크 수: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  [{i}] ({len(c)}자) {c[:60]}...")
    assert len(chunks) >= 2, "청크가 2개 이상이어야 합니다"
    print("  OK 단락 청킹 통과")


# ==============================================================================
# 단위 테스트: pymupdf4llm 정제 + 헤더 청킹
# ==============================================================================

async def test_header_chunking():
    print("\n[2] 헤더 기반 청킹 테스트 (pymupdf4llm 스타일 마크다운)")
    sample = """## 서울중앙병원 이용 안내

본 안내문은 서울중앙병원을 이용하시는 모든 분께 제공됩니다.

## 1. 병원 소개

서울중앙병원은 서울시 강남구 테헤란로 123에 위치합니다.

## ▶ 병원 주요 현황

|구분|내용|
|---|---|
|소재지|서울시 강남구 테헤란로 123|

## 2. 진료 시간

평일 09:00 ~ 17:30, 토요일 09:00 ~ 12:00 운영됩니다.

## ▶ 외래 진료 운영 시간

| 구분 | 접수 | 진료 |
|---|---|---|
| 평일 | 08:30~17:00 | 09:00~17:30 |"""

    cleaned = _clean(sample)
    chunks = _split_sections(cleaned)
    print(f"  청크 수: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  [{i}] ({len(c)}자) {c[:80].replace(chr(10), ' ')}...")
    assert len(chunks) >= 3, "섹션이 3개 이상이어야 합니다"
    print("  OK 헤더 청킹 통과")


# ==============================================================================
# 통합 테스트: PDF → ChromaDB (새 파이프라인)
# ==============================================================================

async def test_pdf_pipeline():
    from app.utils.config import settings
    print(f"\n[3] PDF 파이프라인 테스트 (Mock 임베딩, pymupdf4llm)")

    pdf_path = os.path.join(os.path.dirname(__file__), "hospital_manual.pdf")
    if not os.path.exists(pdf_path):
        print(f"  hospital_manual.pdf 없음 — 건너뜀")
        return None

    import asyncpg
    conn = await asyncpg.connect(settings.database_url)
    try:
        row = await conn.fetchrow(
            "SELECT id FROM tenants WHERE twilio_number = $1", "5"
        )
    finally:
        await conn.close()

    if not row:
        print("  tenant SIP=5 없음 — 건너뜀")
        return None

    tenant_id = str(row["id"])
    print(f"  tenant_id: {tenant_id}")

    processor = PDFProcessor(
        embedder=MockEmbeddingService(),
        rag=ChromaRAGService(),
    )

    doc_id = await processor.process(
        pdf_path=pdf_path,
        tenant_id=tenant_id,
        file_name="hospital_manual.pdf",
        industry="hospital",
    )
    print(f"  document_id: {doc_id}")

    rag = ChromaRAGService()
    import chromadb
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    col = client.get_collection(rag._collection_name(tenant_id))
    count = col.count()
    print(f"  ChromaDB 청크 수: {count}")
    assert count > 0
    print("  OK PDF 파이프라인 통과")
    return tenant_id


# ==============================================================================
# 검색 테스트
# ==============================================================================

async def test_search(tenant_id: str):
    try:
        from app.services.embedding.local import BGEM3LocalEmbeddingService
        embedder = BGEM3LocalEmbeddingService()
        print("\n[검색] BGE-M3 임베딩")
    except Exception:
        embedder = MockEmbeddingService()
        print("\n[검색] Mock 임베딩 (의미 유사도 없음)")

    query = input("  검색할 텍스트를 입력하세요: ").strip()
    if not query:
        print("  입력 없음 — 건너뜀")
        return

    rag = ChromaRAGService()
    results = await rag.search(await embedder.embed(query), tenant_id, top_k=3)
    print(f"\n  검색 결과 ({len(results)}개):")
    for i, chunk in enumerate(results):
        print(f"\n  [{i+1}] {chunk[:120]}{'...' if len(chunk) > 120 else ''}")
    print("\n  OK 검색 테스트 완료")


# ==============================================================================
# 실행
# ==============================================================================

async def main():
    await test_paragraph_chunking()
    await test_header_chunking()
    tenant_id = await test_pdf_pipeline()
    if tenant_id:
        await test_search(tenant_id)
    print("\n모든 테스트 완료")


if __name__ == "__main__":
    asyncio.run(main())
