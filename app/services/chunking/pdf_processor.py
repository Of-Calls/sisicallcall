"""PDF 청킹 파이프라인 — pymupdf4llm + RecursiveCharacterTextSplitter.

변경 이력:
  2026-04-24: pdfplumber → pymupdf4llm. 외부 chunker 주입 + LLM 분류 제거.
              RecursiveCharacterTextSplitter 내장 + 헤더 기반 메타데이터 자동 추출.
  2026-04-27: chunk 별 LLM 메타데이터 보강 (title/summary/keywords/topic) +
              tenant 가용 카테고리 LLM 정제 후 Redis 캐시 (faq_branch RAG miss 안내용).

청킹 전략:
    1차: ## N. 또는 ## ▶ 헤더 기준 섹션 분할 (문서 의미 단위 보존)
    2차: 섹션이 MAX_SECTION_CHARS 초과 시 RCS 추가 분할 (크기 제한)
    3차: chunk 별 GPT-4o-mini batch 호출 → metadata 보강
    4차: tenant 전체 raw topic → 자연어 카테고리 5~7개 정제 → Redis write
"""
import asyncio
import json
import re
import uuid
from typing import Optional

import asyncpg
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.services.embedding.base import BaseEmbeddingService
from app.services.llm.base import BaseLLMService
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.rag.chroma import ChromaRAGService
from app.services.session.redis_session import RedisSessionService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SECTION_SPLIT_RE = re.compile(r'(?=\n## )')   # 모든 ## 헤더를 섹션 경계로
_BOLD_RE = re.compile(r'\*\*([^*]+)\*\*')
_HEADER_META_RE = re.compile(r'^##\s+(?:\d+\.\s*)?(▶\s*|■\s*)?(.+?)$')

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

    진짜 헤더 vs 가짜 헤더(볼드 문장) 구분 기준:
        ** 마크업 포함 → 볼드 문장이 ## 로 오인식된 것 → ## 제거
        ** 마크업 없음  → 실제 섹션 헤더 → 유지
    예:
        ## 찾아오시는 길           (** 없음) → 유지 ✅
        ## 강남구청은 ... **.** ` (** 있음) → ## 제거 ✅
    """
    lines = []
    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('## ') and '**' in s:
            # 볼드 문장 오인식 → ## 제거 후 ** 도 제거해 일반 텍스트로
            lines.append(_BOLD_RE.sub(r'\1', s[3:]))
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


# ── LLM 메타데이터 보강 (Phase 2) ─────────────────────────────────

_CHUNK_ENRICH_BATCH = 10  # batch 10 청크/호출 — 비용/시간/정확도 균형
_JSON_ARRAY_RE = re.compile(r'\[.*\]', re.DOTALL)

_CHUNK_ENRICH_SYSTEM_PROMPT = """당신은 PDF 청크의 메타데이터 추출기입니다.
입력: 청크 N 개 (인덱스 1~N).
출력: JSON 배열, 원소 N 개. 각 원소 필드:
  - title: 청크 핵심 주제 한 줄 (10~25자, 한국어)
  - summary: 청크 요약 1~2문장 (50자 이내)
  - keywords: 검색 키워드 3~5개 (한국어 배열)
  - topic: 카테고리 한 단어 또는 짧은 구 (예: "위치", "예약", "진료시간", "주차", "응급실")

규칙:
- 청크 내용에만 의존. 없는 정보 추측 절대 금지.
- 청크가 모호하거나 짧으면 모든 필드를 짧게 유지.
- 출력은 JSON 배열만, 다른 설명 텍스트 절대 포함하지 않는다."""

_CATEGORY_REFINE_SYSTEM_PROMPT = """당신은 음성 안내용 카테고리 정제기입니다.
입력: chunk 별 raw topic 문자열 list.
출력: 자연스러운 음성 안내용 카테고리 5~7개 (JSON array, 한국어).

규칙:
- 비슷한 의미의 topic 은 통합 (예: "주차장 이용", "주차" → "주차 안내").
- 너무 길거나 모호한 topic 은 제외.
- 음성으로 자연스럽게 들리는 표현 (예: "찾아오시는 길" → "위치 안내").
- 5~7개로 제한.
- JSON array 만 출력, 다른 텍스트 절대 금지."""


def _default_chunk_meta() -> dict:
    return {"title": "", "summary": "", "keywords": [], "topic": "기타"}


async def _enrich_chunks_with_llm(
    chunks: list[str], llm: BaseLLMService
) -> list[dict]:
    """chunks 를 batch 단위로 LLM 호출해 metadata list 반환.

    실패한 batch 는 default 메타로 채워 길이 보장.
    """
    results: list[dict] = []
    for start in range(0, len(chunks), _CHUNK_ENRICH_BATCH):
        batch = chunks[start : start + _CHUNK_ENRICH_BATCH]
        user_msg = "\n\n".join(
            f"[{j + 1}]\n{c}" for j, c in enumerate(batch)
        )
        try:
            raw = await llm.generate(
                system_prompt=_CHUNK_ENRICH_SYSTEM_PROMPT,
                user_message=user_msg,
                temperature=0.1,
                max_tokens=2000,
            )
        except Exception as e:
            logger.error("chunk enrich LLM call failed batch=%d: %s", start // _CHUNK_ENRICH_BATCH, e)
            results.extend([_default_chunk_meta()] * len(batch))
            continue

        match = _JSON_ARRAY_RE.search(raw or "")
        if not match:
            logger.warning(
                "chunk enrich JSON not found batch=%d raw=%r",
                start // _CHUNK_ENRICH_BATCH, (raw or "")[:200],
            )
            results.extend([_default_chunk_meta()] * len(batch))
            continue
        try:
            parsed = json.loads(match.group(0))
        except Exception as e:
            logger.error("chunk enrich JSON parse failed batch=%d: %s", start // _CHUNK_ENRICH_BATCH, e)
            results.extend([_default_chunk_meta()] * len(batch))
            continue

        if not isinstance(parsed, list):
            results.extend([_default_chunk_meta()] * len(batch))
            continue

        # 길이 맞추기 — LLM 이 N 개 안 맞춰 줄 수 있음
        normalized: list[dict] = []
        for item in parsed[: len(batch)]:
            if isinstance(item, dict):
                normalized.append({
                    "title": str(item.get("title", ""))[:100],
                    "summary": str(item.get("summary", ""))[:300],
                    "keywords": [str(k) for k in (item.get("keywords") or []) if k][:5],
                    "topic": str(item.get("topic", "기타"))[:50],
                })
            else:
                normalized.append(_default_chunk_meta())
        while len(normalized) < len(batch):
            normalized.append(_default_chunk_meta())
        results.extend(normalized)

    return results


async def _refine_categories(topics: list[str], llm: BaseLLMService) -> list[str]:
    """raw topic list → 자연스러운 5~7개 카테고리. LLM 1회 호출."""
    distinct = sorted({t.strip() for t in topics if t and t.strip() and t != "기타"})
    if not distinct:
        return []

    try:
        raw = await llm.generate(
            system_prompt=_CATEGORY_REFINE_SYSTEM_PROMPT,
            user_message=f"raw topics: {distinct}",
            temperature=0.1,
            max_tokens=300,
        )
    except Exception as e:
        logger.error("category refine LLM call failed: %s", e)
        return distinct[:7]

    match = _JSON_ARRAY_RE.search(raw or "")
    if not match:
        logger.warning("category refine JSON not found raw=%r", (raw or "")[:200])
        return distinct[:7]
    try:
        parsed = json.loads(match.group(0))
    except Exception as e:
        logger.error("category refine JSON parse failed: %s", e)
        return distinct[:7]

    if not isinstance(parsed, list):
        return distinct[:7]
    return [str(c).strip() for c in parsed if c][:7]


class PDFProcessor:
    def __init__(
        self,
        embedder: BaseEmbeddingService,
        rag: ChromaRAGService,
        llm: Optional[BaseLLMService] = None,
        session: Optional[RedisSessionService] = None,
    ):
        self._embedder = embedder
        self._rag = rag
        # LLM 메타데이터 보강용 — 미주입 시 GPT-4o-mini default
        self._llm = llm or GPT4OMiniService()
        # tenant rag_categories Redis write 용 — 미주입 시 default 인스턴스
        self._session = session or RedisSessionService()

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

            # LLM 메타데이터 보강 — chunk 별 title/summary/keywords/topic
            llm_metas = await _enrich_chunks_with_llm(chunks, self._llm)
            logger.info("llm enrich done doc_id=%s metas=%d", document_id, len(llm_metas))

            collection_name = self._rag._collection_name(tenant_id)
            for i, (chunk, embedding, llm_meta) in enumerate(
                zip(chunks, embeddings, llm_metas)
            ):
                meta = _metadata_from_chunk(chunk)
                # ChromaDB metadata 는 primitive 만 → keywords list 는 콤마 join
                keywords_str = ", ".join(llm_meta.get("keywords") or [])[:200]
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
                        # LLM 보강 메타데이터 (Phase 2)
                        "llm_title": llm_meta.get("title", ""),
                        "llm_summary": llm_meta.get("summary", ""),
                        "llm_keywords": keywords_str,
                        "llm_topic": llm_meta.get("topic", "기타"),
                    },
                )

            # tenant 가용 카테고리 LLM 정제 + Redis write
            topics = [m.get("topic", "") for m in llm_metas]
            refined_categories = await _refine_categories(topics, self._llm)
            if refined_categories:
                await self._session.set_rag_categories(tenant_id, refined_categories)
                logger.info(
                    "rag_categories refined tenant=%s categories=%s",
                    tenant_id, refined_categories,
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
