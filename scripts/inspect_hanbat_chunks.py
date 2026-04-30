"""한밭식당 ChromaDB 청크 전체 내용 출력 → tmp/hanbat_chunks.txt"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import chromadb
from app.utils.config import settings

client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
col = client.get_collection("tenant_ba2bf4996fcc4340b3dd9341f8bcc915_docs")
r = col.get(include=["documents", "metadatas"])

out = ROOT / "tmp" / "hanbat_chunks.txt"
out.parent.mkdir(exist_ok=True)

with out.open("w", encoding="utf-8") as f:
    for i, (cid, doc, meta) in enumerate(zip(r["ids"], r["documents"], r["metadatas"])):
        f.write(f"\n{'='*60}\n")
        f.write(f"[{i}] {cid}\n")
        f.write(f"is_auth: {meta.get('is_auth')}  topic: {meta.get('topic','?')}  title: {meta.get('title','?')}\n")
        f.write("---\n")
        f.write((doc or "")[:800] + "\n")

print(f"출력 완료: {out}")
