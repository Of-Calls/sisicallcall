"""3개 tenant PDF 재인덱싱 스크립트.

실행 전:
    .env 에 OPENAI_API_KEY, DATABASE_URL, CHROMA_HOST 등 설정 확인.

실행:
    python scripts/reindex_all_tenants.py

작업 순서:
    1. 한밭식당 SIP → '3' 업데이트
    2. 강남구청 tenant 생성 (SIP='4')
    3. 병원 tenant_id 조회
    4. ChromaDB 기존 컬렉션 삭제
    5. rag_documents 기존 레코드 삭제
    6. PDFProcessor 로 3개 PDF 순차 인덱싱
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import asyncpg
import chromadb

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

PDFS = {
    "hospital": {
        "path": ROOT / "tests" / "hospital_manual.pdf",
        "industry": "hospital",
        "sip": "5",
    },
    "district_office": {
        "path": ROOT / "tests" / "district_office_manual.pdf",
        "industry": "government",
        "sip": "4",
    },
    "store": {
        "path": ROOT / "tests" / "store_maual.pdf",  # 오타 파일명 그대로 사용
        "industry": "restaurant",
        "sip": "3",
    },
}


async def setup_tenants(conn) -> dict[str, str]:
    """tenant DB 정리 후 {sip: tenant_id} 매핑 반환."""

    # 한밭식당 SIP → '3'
    await conn.execute(
        "UPDATE tenants SET twilio_number = $1 WHERE name = '한밭식당'",
        "3",
    )
    logger.info("한밭식당 SIP → '3' 업데이트")

    # 강남구청 upsert (이미 있으면 SIP만 보정)
    existing = await conn.fetchrow(
        "SELECT id FROM tenants WHERE twilio_number = '4'"
    )
    if not existing:
        row = await conn.fetchrow(
            """
            INSERT INTO tenants (name, twilio_number, industry, settings)
            VALUES ('강남구청', '4', 'government', '{}')
            RETURNING id
            """,
        )
        logger.info("강남구청 tenant 생성 id=%s", row["id"])
    else:
        # 이름 보정
        await conn.execute(
            "UPDATE tenants SET name = '강남구청' WHERE twilio_number = '4'"
        )
        logger.info("강남구청 tenant 이미 존재 — name 보정")

    # 전체 테넌트 조회
    rows = await conn.fetch("SELECT id, name, twilio_number FROM tenants")
    mapping = {}
    for r in rows:
        mapping[r["twilio_number"]] = str(r["id"])
        logger.info("tenant: sip=%s  name=%s  id=%s", r["twilio_number"], r["name"], r["id"])

    return mapping


async def clear_chroma(tenant_ids: list[str]) -> None:
    """ChromaDB 기존 컬렉션 삭제."""
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    existing = {c.name for c in client.list_collections()}

    for tid in tenant_ids:
        col_name = f"tenant_{tid.replace('-', '')}_docs"
        if col_name in existing:
            client.delete_collection(col_name)
            logger.info("ChromaDB 컬렉션 삭제: %s", col_name)
        else:
            logger.info("ChromaDB 컬렉션 없음 (skip): %s", col_name)


async def clear_rag_documents(conn, tenant_ids: list[str]) -> None:
    """rag_documents 기존 레코드 삭제."""
    for tid in tenant_ids:
        deleted = await conn.execute(
            "DELETE FROM rag_documents WHERE tenant_id = $1::uuid",
            tid,
        )
        logger.info("rag_documents 삭제 tenant=%s  result=%s", tid[:8], deleted)


async def run_indexing(tenant_id: str, pdf_path: Path, industry: str) -> None:
    """단일 PDF 인덱싱."""
    from app.services.embedding.local import BGEM3LocalEmbeddingService
    from app.services.rag.chroma import ChromaRAGService
    from app.services.chunking.pdf_processor import PDFProcessor

    embedder = BGEM3LocalEmbeddingService()
    rag = ChromaRAGService()
    processor = PDFProcessor(embedder=embedder, rag=rag)

    logger.info("인덱싱 시작 tenant=%s  pdf=%s", tenant_id[:8], pdf_path.name)
    doc_id = await processor.process(
        pdf_path=str(pdf_path),
        tenant_id=tenant_id,
        file_name=pdf_path.name,
        industry=industry,
    )
    logger.info("인덱싱 완료 tenant=%s  doc_id=%s", tenant_id[:8], doc_id)


async def main():
    # PDF 파일 존재 확인
    for key, info in PDFS.items():
        if not info["path"].exists():
            logger.error("PDF 없음: %s", info["path"])
            sys.exit(1)
    logger.info("PDF 3개 존재 확인 완료")

    conn = await asyncpg.connect(settings.database_url)
    try:
        # 1. Tenant 정리
        sip_to_id = await setup_tenants(conn)

        tenant_ids = [sip_to_id[info["sip"]] for info in PDFS.values()]

        # 2. ChromaDB 초기화
        await clear_chroma(tenant_ids)

        # 3. rag_documents 초기화
        await clear_rag_documents(conn, tenant_ids)

    finally:
        await conn.close()

    # 4. 순차 인덱싱 (BGE-M3 GPU 메모리 고려)
    for key, info in PDFS.items():
        tenant_id = sip_to_id[info["sip"]]
        await run_indexing(
            tenant_id=tenant_id,
            pdf_path=info["path"],
            industry=info["industry"],
        )

    logger.info("=" * 50)
    logger.info("전체 재인덱싱 완료")

    # 5. 결과 확인
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    for key, info in PDFS.items():
        tid = sip_to_id[info["sip"]]
        col_name = f"tenant_{tid.replace('-', '')}_docs"
        try:
            col = client.get_collection(col_name)
            logger.info("  [%s] %s → %d chunks", key, col_name[:30], col.count())
        except Exception:
            logger.warning("  [%s] 컬렉션 조회 실패", key)


if __name__ == "__main__":
    asyncio.run(main())
