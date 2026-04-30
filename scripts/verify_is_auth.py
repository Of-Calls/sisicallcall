"""ChromaDB is_auth 설정 검증 스크립트."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import chromadb
from app.utils.config import settings

client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)

COLLECTIONS = {
    "한밭식당":     "tenant_ba2bf4996fcc4340b3dd9341f8bcc915_docs",
    "서울중앙병원": "tenant_05665d1ae25b44f7be6ca262603dbfd5_docs",
    "강남구청":     "tenant_580cd81157a94e50aec51d99d05915b2_docs",
}

for name, col_name in COLLECTIONS.items():
    col = client.get_collection(col_name)
    r = col.get(include=["metadatas"])
    metas = r["metadatas"]
    true_cnt  = sum(1 for m in metas if m.get("is_auth") is True)
    false_cnt = sum(1 for m in metas if m.get("is_auth") is False)
    none_cnt  = sum(1 for m in metas if "is_auth" not in m)
    total = len(r["ids"])
    print(f"{name}: total={total}  is_auth=True:{true_cnt}  False:{false_cnt}  missing:{none_cnt}")
