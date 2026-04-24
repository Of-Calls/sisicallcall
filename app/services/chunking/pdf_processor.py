"""PDF 청킹 파이프라인 — pymupdf4llm + RecursiveCharacterTextSplitter.

변경 이력 2026-04-24:
    - pdfplumber (구조 손실) → pymupdf4llm (마크다운 헤더/표 보존)
    - 외부 chunker 주입 + LLM 분류 제거
    - RecursiveCharacterTextSplitter 내장 + 헤더 기반 메타데이터 자동 추출

청킹 전략:
    1차: ## N. 또는 ## ▶ 헤더 기준 섹션 분할 (문서 의미 단위 보존)
    2차: 섹션이 MAX_SECTION_CHARS 초과 시 RCS 추가 분할 (크기 제한)
"""
import asyncio
import re
import uuid

import asyncpg
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.services.embedding.base import BaseEmbeddingService
from app.services.rag.chroma import ChromaRAGService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SECTION_SPLIT_RE = re.compile(r'(?=\n## (?:\d+\.|▶))')
_BOLD_RE = re.compile(r'\*\*([^*]+)\*\*')
_HEADER_META_RE = re.compile(r'^##\s+(?:\d+\.\s*)?(▶\s*)?(.+?)$')

MAX_SECTION_CHARS = 1000

_RCS = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n■", "\n▶", "\n☑", "\n※", "\n", " "],
    chunk_size=MAX_SECTION_CHARS,
    chunk_overlap=100,
)


def _extract_text(pdf_path: str) -> str:
    """pymupdf4llm 로 PDF → 마크다운 텍스트 추출."""
    import pymupdf4llm
    return pymupdf4llm.to_markdown(pdf_path)


def _clean(text: str) -> str:
    """pymupdf4llm 마크다운 아티팩트 정리.

    볼드 문장이 ## 로 오인식되는 케이스를 필터링하고
    ** 마크업을 제거해 임베딩 노이즈를 줄인다.
    진짜 헤더 패턴: ## N. Title 또는 ## ▶ Title
    """
    lines = []
    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('## ') and not re.match(r'^##\s+(\d+\.|▶)', s):
            lines.append(s[3:])  # ## 제거 → 일반 텍스트
        else:
            lines.append(line)
    cleaned = '\n'.join(lines)
    return _BOLD_RE.sub(r'\1', cleaned)


def _split_sections(text: str) -> list[str]:
    """헤더 기준 섹션 분할 → 긴 섹션은 RCS 추가 분할."""
    raw = _SECTION_SPLIT_RE.split(text)
    chunks = []
    for section in raw:
        section = section.strip()
        if not section:
            continue
        if len(section) <= MAX_SECTION_CHARS:
            chunks.append(section)
        else:
            sub = _RCS.split_text(section)
            chunks.extend(c.strip() for c in sub if c.strip())
    return chunks


def _metadata_from_chunk(chunk: str) -> dict:
    """청크 상단 헤더에서 category / product_name 자동 추출."""
    for line in chunk.split('\n')[:5]:
        m = _HEADER_META_RE.match(line.strip())
        if m:
            title = m.group(2).strip()
            return {"category": title, "product_name": title}
    first = next(
        (l.strip() for l in chunk.split('\n') if l.strip() and not l.startswith('#')),
        "",
    )
    return {"category": "기타", "product_name": first[:60]}


class PDFProcessor:
    def __init__(
        self,
        embedder: BaseEmbeddingService,
        rag: ChromaRAGService,
    ):
        self._embedder = embedder
        self._rag = rag

    async def process(
        self,
        pdf_path: str,
        tenant_id: str,
        file_name: str,
        industry: str,
    ) -> str:
        """PDF → 청킹 → 임베딩 → ChromaDB 저장. document_id 반환."""
        existing = await self._find_existing_document(tenant_id, file_name)
        if existing:
            logger.info("duplicate detected, replacing doc_id=%s file=%s", existing["id"], file_name)
            await self._rag.delete_by_document(str(existing["id"]), tenant_id)
            await self._delete_rag_document(existing["id"])

        document_id = await self._insert_rag_document(tenant_id, file_name)
        logger.info("rag_document created id=%s file=%s", document_id, file_name)

        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, _extract_text, pdf_path)
            text = _clean(raw)
            logger.info("pdf extracted+cleaned len=%d file=%s", len(text), file_name)

            chunks = _split_sections(text)
            logger.info("chunked count=%d file=%s", len(chunks), file_name)

            embeddings = await self._embedder.embed_batch(chunks)

            collection_name = self._rag._collection_name(tenant_id)
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                meta = _metadata_from_chunk(chunk)
                await self._rag.upsert(
                    doc_id=f"{document_id}_chunk_{i}",
                    content=chunk,
                    embedding=embedding,
                    tenant_id=tenant_id,
                    metadata={
                        "tenant_id": tenant_id,
                        "document_id": str(document_id),
                        "file_name": file_name,
                        "chunk_index": i,
                        "product_name": meta["product_name"],
                        "category": meta["category"],
                        "industry": industry,
                    },
                )

            await self._update_rag_document(document_id, len(chunks), collection_name)
            logger.info("pdf_processor done doc_id=%s chunks=%d", document_id, len(chunks))

        except Exception as e:
            await self._fail_rag_document(document_id)
            logger.error("pdf_processor failed doc_id=%s err=%s", document_id, e)
            raise

        return str(document_id)

    async def delete_document(self, document_id: str, tenant_id: str) -> None:
        """문서 삭제 — ChromaDB 청크 + rag_documents 동시 삭제."""
        await self._rag.delete_by_document(document_id, tenant_id)
        await self._delete_rag_document(uuid.UUID(document_id))
        logger.info("document deleted doc_id=%s tenant=%s", document_id, tenant_id)

    async def _insert_rag_document(self, tenant_id: str, file_name: str) -> uuid.UUID:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO rag_documents (tenant_id, file_name, file_type, status)
                VALUES ($1::uuid, $2, 'pdf', 'processing')
                RETURNING id
                """,
                tenant_id,
                file_name,
            )
            return row["id"]
        finally:
            await conn.close()

    async def _update_rag_document(
        self, document_id: uuid.UUID, chunk_count: int, collection_name: str
    ) -> None:
        conn = await asyncpg.connect(settings.database_url)
        try:
            await conn.execute(
                """
                UPDATE rag_documents
                SET status = 'ready',
                    chunk_count = $2,
                    chroma_collection = $3,
                    indexed_at = now()
                WHERE id = $1
                """,
                document_id,
                chunk_count,
                collection_name,
            )
        finally:
            await conn.close()

    async def _fail_rag_document(self, document_id: uuid.UUID) -> None:
        conn = await asyncpg.connect(settings.database_url)
        try:
            await conn.execute(
                "UPDATE rag_documents SET status = 'failed' WHERE id = $1",
                document_id,
            )
        finally:
            await conn.close()

    async def _find_existing_document(self, tenant_id: str, file_name: str) -> dict | None:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                """
                SELECT id FROM rag_documents
                WHERE tenant_id = $1::uuid AND file_name = $2 AND status != 'failed'
                ORDER BY uploaded_at DESC LIMIT 1
                """,
                tenant_id,
                file_name,
            )
            return dict(row) if row else None
        finally:
            await conn.close()

    async def _delete_rag_document(self, document_id: uuid.UUID) -> None:
        conn = await asyncpg.connect(settings.database_url)
        try:
            await conn.execute("DELETE FROM rag_documents WHERE id = $1", document_id)
        finally:
            await conn.close()
