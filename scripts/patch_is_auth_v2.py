"""한밭식당 is_auth 확장 패치 — 예약 절차 관련 청크 전수 커버.

조건 (합집합):
  1. 명시 IDs (patch_is_auth.py 기존 5개 — 예약 절차 핵심 청크)
  2. llm_topic 또는 llm_title에 '예약' 포함 (키워드 확장 — 로그에서 is_auth=False top-1 보완)
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

# 예약 절차 핵심 청크 — 명시 세트 (이전 patch_is_auth.py 기준)
_AUTH_EXPLICIT_IDS = {
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
    topic = meta.get("llm_topic", "") or ""
    title = meta.get("llm_title", "") or ""
    # 합집합: 명시 세트 OR '예약' 키워드 포함
    if cid in _AUTH_EXPLICIT_IDS or "예약" in topic or "예약" in title:
        true_ids.append(cid)
        true_metas.append({**meta, "is_auth": True})
    else:
        false_ids.append(cid)
        false_metas.append({**meta, "is_auth": False})

if true_ids:
    col.update(ids=true_ids, metadatas=true_metas)
if false_ids:
    col.update(ids=false_ids, metadatas=false_metas)

r2 = col.get(include=["metadatas"])
true_cnt = sum(1 for m in r2["metadatas"] if m.get("is_auth") is True)
false_cnt = sum(1 for m in r2["metadatas"] if m.get("is_auth") is False)
print(f"완료: is_auth=True:{true_cnt}  False:{false_cnt}  total:{len(r2['ids'])}")
print("is_auth=True 청크:")
for cid, meta in zip(r2["ids"], r2["metadatas"]):
    if meta.get("is_auth") is True:
        topic_val = meta.get("llm_topic", "")
        title_val = (meta.get("llm_title", "") or "")[:40]
        print(f"  {cid}  topic={topic_val!r}  title={title_val!r}")
