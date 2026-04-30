"""ChromaDB is_auth 필드 설정 스크립트.

동작:
  1. PostgreSQL 에서 전체 테넌트 ID 조회
  2. ChromaDB 전체 컬렉션 열거
  3. 한밭식당 컬렉션: 예약 관련 청크 → is_auth=True
  4. 나머지 모든 청크 → is_auth=False (명시적 설정)

실행:
    python scripts/set_is_auth.py [--dry-run]

--dry-run: ChromaDB 업데이트 없이 대상 청크만 출력
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

DRY_RUN = "--dry-run" in sys.argv

# 예약 관련 키워드 (한밭식당 청크 검색용)
RESERVATION_KEYWORDS = [
    "예약", "reservation", "booking", "예약하", "예약 가능", "예약 문의",
    "예약 접수", "예약금", "테이블", "자리", "좌석",
]


def is_reservation_chunk(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in RESERVATION_KEYWORDS)


async def get_tenants(conn) -> dict[str, str]:
    """sip_number → tenant_id 매핑 반환."""
    rows = await conn.fetch("SELECT id, name, twilio_number FROM tenants ORDER BY name")
    result = {}
    for r in rows:
        result[str(r["twilio_number"])] = {"id": str(r["id"]), "name": r["name"]}
        logger.info("tenant: name=%s  sip=%s  id=%s", r["name"], r["twilio_number"], r["id"])
    return result


def collection_name(tenant_id: str) -> str:
    return f"tenant_{tenant_id.replace('-', '')}_docs"


async def process_collection(
    col: chromadb.Collection,
    is_hanbat: bool,
) -> tuple[int, int]:
    """컬렉션 내 청크를 처리. (auth_set, false_set) 카운트 반환."""
    result = col.get(include=["documents", "metadatas"])
    ids = result["ids"]
    docs = result["documents"]
    metas = result["metadatas"]

    if not ids:
        logger.info("  컬렉션 비어있음: %s", col.name)
        return 0, 0

    auth_ids, auth_metas = [], []
    false_ids, false_metas = [], []

    for chunk_id, doc, meta in zip(ids, docs, metas):
        meta = meta or {}
        if is_hanbat and is_reservation_chunk(doc or ""):
            new_meta = {**meta, "is_auth": True}
            auth_ids.append(chunk_id)
            auth_metas.append(new_meta)
        else:
            current = meta.get("is_auth")
            if current is not True and str(current).lower() != "true":
                new_meta = {**meta, "is_auth": False}
                false_ids.append(chunk_id)
                false_metas.append(new_meta)

    # is_auth=True 대상 출력
    if auth_ids:
        logger.info("  [is_auth=True 대상] %d 청크", len(auth_ids))
        for cid in auth_ids:
            logger.info("    → %s", cid)

    if DRY_RUN:
        logger.info("  [DRY-RUN] 실제 업데이트 생략")
        return len(auth_ids), len(false_ids)

    if auth_ids:
        col.update(ids=auth_ids, metadatas=auth_metas)
        logger.info("  is_auth=True 업데이트 완료: %d 청크", len(auth_ids))

    if false_ids:
        col.update(ids=false_ids, metadatas=false_metas)
        logger.info("  is_auth=False 업데이트 완료: %d 청크", len(false_ids))

    return len(auth_ids), len(false_ids)


async def main():
    conn = await asyncpg.connect(settings.database_url)
    try:
        tenants = await get_tenants(conn)
    finally:
        await conn.close()

    # 한밭식당 (SIP=3) tenant_id
    hanbat_info = tenants.get("3")
    if not hanbat_info:
        logger.error("한밭식당(SIP=3) 테넌트를 DB에서 찾지 못했습니다.")
        sys.exit(1)
    hanbat_id = hanbat_info["id"]
    hanbat_col_name = collection_name(hanbat_id)
    logger.info("한밭식당 tenant_id=%s  collection=%s", hanbat_id, hanbat_col_name)

    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    collections = client.list_collections()
    logger.info("\n전체 ChromaDB 컬렉션: %d 개", len(collections))
    for c in collections:
        logger.info("  %s (%d 청크)", c.name, c.count())

    total_auth = 0
    total_false = 0

    for col_meta in collections:
        col = client.get_collection(col_meta.name)
        is_hanbat = col_meta.name == hanbat_col_name
        label = "한밭식당 ★" if is_hanbat else col_meta.name
        logger.info("\n처리 중: %s", label)
        a, f = await process_collection(col, is_hanbat=is_hanbat)
        total_auth += a
        total_false += f

    logger.info("\n=== 완료 ===")
    logger.info("is_auth=True 설정: %d 청크", total_auth)
    logger.info("is_auth=False 설정: %d 청크", total_false)
    if DRY_RUN:
        logger.info("[DRY-RUN 모드 — 실제 변경 없음]")


if __name__ == "__main__":
    asyncio.run(main())
