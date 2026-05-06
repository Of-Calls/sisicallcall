"""Chroma RAG top-k 검색 결과를 터미널에 출력 (FAQ 분기와 동일 경로).

    python -m scripts.rag_peek --tenant-id <uuid> --query "병원 위치 어디예요" -k 5

컬렉션 전체 청크 나열(임베딩 검색 없음):

    python -m scripts.rag_peek --tenant-id <uuid> --list-all --max 20
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

import chromadb

from app.services.embedding import get_embedder
from app.services.rag.chroma import ChromaRAGService
from app.utils.config import settings


def _collection_name(tenant_id: str) -> str:
    return f"tenant_{tenant_id.replace('-', '')}_docs"


async def _peek_search(tenant_id: str, query: str, top_k: int, threshold: float) -> None:
    embedder = get_embedder()
    rag = ChromaRAGService()
    emb = await embedder.embed(query)
    rows = await rag.search_with_meta(emb, tenant_id, top_k=top_k)
    print(f"tenant_id={tenant_id}")
    print(f"collection={_collection_name(tenant_id)}")
    print(f"query={query!r}  top_k={top_k}  faq_threshold(참고)={threshold}\n")
    for i, r in enumerate(rows):
        dist = r.get("distance")
        meta = r.get("metadata") or {}
        ok = dist is not None and dist <= threshold
        title = meta.get("llm_title") or meta.get("title") or ""
        print(f"--- [{i + 1}] distance={dist}  threshold_pass={ok}")
        print(f"    id={r.get('id')!r}  title={title!r}")
        print(f"    is_auth={meta.get('is_auth')}  is_vision={meta.get('is_vision')}")
        doc = r.get("document") or ""
        print(doc)
        print()


def _peek_list_all(tenant_id: str, max_rows: int) -> None:
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    name = _collection_name(tenant_id)
    col = client.get_collection(name)
    n = col.count()
    r = col.get(include=["documents", "metadatas"], limit=min(max_rows, n) if n else 0)
    print(f"collection={name}  total_count={n}  showing={len(r.get('ids') or [])}\n")
    ids = r.get("ids") or []
    docs = r.get("documents") or []
    metas = r.get("metadatas") or []
    for i, (cid, doc, meta) in enumerate(zip(ids, docs, metas)):
        meta = meta or {}
        title = meta.get("llm_title") or meta.get("title") or ""
        print(f"--- [{i + 1}] id={cid!r}  title={title!r}")
        print(doc or "")
        print()


def main() -> None:
    p = argparse.ArgumentParser(description="Chroma RAG 청크 / top-k 검색 미리보기")
    p.add_argument("--tenant-id", required=True, help="tenant UUID")
    p.add_argument("--query", default="", help="검색 문장 (FAQ와 동일 임베딩)")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="FAQ _DIST_THRESHOLD 와 비교용 (출력만, 필터는 적용 안 함)",
    )
    p.add_argument(
        "--list-all",
        action="store_true",
        help="유사도 검색 없이 컬렉션에서 앞쪽 청크만 나열",
    )
    p.add_argument(
        "--max",
        type=int,
        default=50,
        dest="max_rows",
        help="--list-all 일 때 최대 개수",
    )
    args = p.parse_args()

    if args.list_all:
        _peek_list_all(args.tenant_id, args.max_rows)
        return

    if not args.query.strip():
        print("--query 가 필요합니다. (또는 --list-all)", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_peek_search(args.tenant_id, args.query.strip(), args.top_k, args.threshold))


if __name__ == "__main__":
    main()
