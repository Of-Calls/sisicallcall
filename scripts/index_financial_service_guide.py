"""app/pdf_files/financial_service_guide.pdf 단일 인덱싱.

reseed_one_tenant.py 와 동일한 PDFProcessor 파이프라인(청킹 → polish → Chroma + rag_documents).

실행 (기본: 금융 가이드 전용 tenant 자동 생성/조회 후 인덱싱):
    python scripts/index_financial_service_guide.py

기존 tenant 에 붙이기:
    python scripts/index_financial_service_guide.py --tenant-id <uuid>

해당 tenant 의 Chroma 컬렉션 + rag_documents 전부 비운 뒤 인덱싱 (reseed 와 동일):
    python scripts/index_financial_service_guide.py --wipe
"""
from __future__ import annotations

import argparse
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

PDF_REL = Path("app") / "pdf_files" / "financial_service_guide.pdf"
DEFAULT_TWILIO = "+821000000090"
DEFAULT_TENANT_NAME = "금융서비스가이드"
INDUSTRY = "finance"


async def _ensure_tenant(conn: asyncpg.Connection) -> str:
    row = await conn.fetchrow(
        "SELECT id FROM tenants WHERE twilio_number = $1",
        DEFAULT_TWILIO,
    )
    if row:
        return str(row["id"])
    row = await conn.fetchrow(
        """
        INSERT INTO tenants (name, twilio_number, industry, plan, settings)
        VALUES ($1, $2, $3, 'basic', '{}'::jsonb)
        RETURNING id
        """,
        DEFAULT_TENANT_NAME,
        DEFAULT_TWILIO,
        INDUSTRY,
    )
    logger.info(
        "tenant 생성 name=%s twilio=%s id=%s",
        DEFAULT_TENANT_NAME,
        DEFAULT_TWILIO,
        str(row["id"])[:8],
    )
    return str(row["id"])


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tenant-id",
        default=None,
        help="기존 tenant UUID. 미지정 시 twilio=%s 인 tenant 를 찾거나 생성합니다."
        % DEFAULT_TWILIO,
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="인덱싱 전 해당 tenant 의 Chroma 컬렉션 + rag_documents 전체 삭제",
    )
    args = parser.parse_args()

    pdf_path = ROOT / PDF_REL
    if not pdf_path.exists():
        logger.error("PDF 없음: %s", pdf_path)
        sys.exit(1)

    conn = await asyncpg.connect(settings.database_url)
    try:
        if args.tenant_id:
            tenant_id = args.tenant_id.strip()
            row = await conn.fetchrow(
                "SELECT id, name FROM tenants WHERE id = $1::uuid",
                tenant_id,
            )
            if not row:
                logger.error("tenant 없음 id=%s", tenant_id)
                sys.exit(1)
            logger.info(
                "tenant=%s (%s)", str(row["id"])[:8], row["name"],
            )
        else:
            tenant_id = await _ensure_tenant(conn)
            logger.info("tenant_id=%s (ensure)", tenant_id[:8])
    finally:
        await conn.close()

    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    col_name = f"tenant_{tenant_id.replace('-', '')}_docs"

    if args.wipe:
        try:
            client.delete_collection(col_name)
            logger.info("chroma 컬렉션 삭제: %s", col_name)
        except Exception as e:
            logger.info("chroma 컬렉션 없음 (skip): %s — %s", col_name, e)

        conn = await asyncpg.connect(settings.database_url)
        try:
            deleted = await conn.execute(
                "DELETE FROM rag_documents WHERE tenant_id = $1::uuid",
                tenant_id,
            )
            logger.info("rag_documents 삭제 result=%s", deleted)
        finally:
            await conn.close()

    from app.services.embedding import get_embedder
    from app.services.rag.chroma import ChromaRAGService
    from app.services.chunking.pdf_processor import PDFProcessor

    embedder = get_embedder()
    logger.info(
        "embedder=%s (provider=%s)",
        type(embedder).__name__,
        settings.embedding_provider,
    )
    rag = ChromaRAGService()
    processor = PDFProcessor(embedder=embedder, rag=rag)

    doc_id = await processor.process(
        pdf_path=str(pdf_path),
        tenant_id=tenant_id,
        file_name=pdf_path.name,
        industry=INDUSTRY,
    )
    logger.info("인덱싱 완료 doc_id=%s", doc_id)

    col = client.get_collection(col_name)
    logger.info("chroma 컬렉션=%s 청크 수=%d", col_name, col.count())


if __name__ == "__main__":
    asyncio.run(main())
