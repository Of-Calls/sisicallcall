"""한밭식당 is_auth 정밀 패치.

예약 절차가 핵심 내용인 청크 5개만 is_auth=True,
나머지는 모두 is_auth=False.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import chromadb
from app.utils.config import settings

HANBAT_COL = "tenant_ba2bf4996fcc4340b3dd9341f8bcc915_docs"

# 예약 절차가 핵심 내용인 청크만 is_auth=True
AUTH_TRUE_IDS = {
    "99658eff-10d0-41e5-bc5d-129cc7ddbee3_chunk_1",   # 방문 전 확인사항 (예약 후 방문)
    "99658eff-10d0-41e5-bc5d-129cc7ddbee3_chunk_5",   # 예약 방법 (전화·온라인·단체)
    "99658eff-10d0-41e5-bc5d-129cc7ddbee3_chunk_6",   # 예약 변경·취소
    "99658eff-10d0-41e5-bc5d-129cc7ddbee3_chunk_11",  # 단체·행사 예약 절차
    "99658eff-10d0-41e5-bc5d-129cc7ddbee3_chunk_12",  # 단체 행사 세부 (예약금)
}

client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
col = client.get_collection(HANBAT_COL)
r = col.get(include=["metadatas"])

true_ids, true_metas = [], []
false_ids, false_metas = [], []

for cid, meta in zip(r["ids"], r["metadatas"]):
    meta = meta or {}
    if cid in AUTH_TRUE_IDS:
        true_ids.append(cid)
        true_metas.append({**meta, "is_auth": True})
    else:
        false_ids.append(cid)
        false_metas.append({**meta, "is_auth": False})

col.update(ids=true_ids, metadatas=true_metas)
col.update(ids=false_ids, metadatas=false_metas)

# 결과 확인
r2 = col.get(include=["metadatas"])
true_cnt  = sum(1 for m in r2["metadatas"] if m.get("is_auth") is True)
false_cnt = sum(1 for m in r2["metadatas"] if m.get("is_auth") is False)
print(f"완료: is_auth=True:{true_cnt}  False:{false_cnt}  total:{len(r2['ids'])}")
print("is_auth=True 청크:")
for cid, meta in zip(r2["ids"], r2["metadatas"]):
    if meta.get("is_auth") is True:
        print(f"  {cid}")
