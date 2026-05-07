"""ChromaDB 컬렉션별 chunk 전수 덤프 — 청킹 경계 검증용 일회성 스크립트."""

import sys, io, os, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import chromadb

INDUSTRY_TO_FILENAME = {
    "hospital": "chunks_hospital.md",
    "government": "chunks_district.md",
    "restaurant": "chunks_restaurant.md",
    "finance": "chunks_finance.md",
}

OUT_DIR = os.path.join(os.path.dirname(__file__))


def short(text: str, n: int = 40) -> str:
    """본문 head/tail 요약용 — 줄바꿈 / 공백 squish."""
    if not text:
        return ""
    s = " ".join(text.split())
    return s[:n]


def dump_collection(col):
    rows = col.get(include=["documents", "metadatas"])
    items = list(zip(rows["ids"], rows["documents"], rows["metadatas"]))

    items.sort(key=lambda x: (x[2] or {}).get("chunk_index", 0))

    industry = (items[0][2] or {}).get("industry", "unknown") if items else "unknown"
    fname = INDUSTRY_TO_FILENAME.get(industry, f"chunks_{industry}.md")
    path = os.path.join(OUT_DIR, fname)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {col.name}\n\n")
        f.write(f"- industry: `{industry}`\n")
        f.write(f"- 총 chunk: **{len(items)}**\n\n")

        f.write("## TOC (요약 한 줄)\n\n")
        f.write("| idx | chars | title | head 40자 | tail 40자 |\n")
        f.write("|---:|---:|---|---|---|\n")
        for _id, doc, meta in items:
            meta = meta or {}
            idx = meta.get("chunk_index", "?")
            title = (meta.get("llm_title") or meta.get("category") or "").replace(
                "|", "/"
            )
            head = short(doc, 40).replace("|", "/")
            tail_src = doc[-40:] if doc else ""
            tail = short(tail_src, 40).replace("|", "/")
            f.write(
                f"| {idx} | {len(doc) if doc else 0} | {title} | {head} | {tail} |\n"
            )

        f.write("\n---\n\n## 전체 본문\n\n")
        for _id, doc, meta in items:
            meta = meta or {}
            idx = meta.get("chunk_index", "?")
            f.write(f"### chunk #{idx}  ·  {len(doc) if doc else 0} chars\n\n")
            f.write("**meta**\n\n")
            f.write("```json\n")
            f.write(json.dumps(meta, ensure_ascii=False, indent=2))
            f.write("\n```\n\n")
            f.write("**body**\n\n")
            f.write("```\n")
            f.write(doc or "")
            f.write("\n```\n\n---\n\n")

    print(f"wrote {path}  ({len(items)} chunks, industry={industry})")


def main():
    client = chromadb.HttpClient(host="localhost", port=8001)
    for c in client.list_collections():
        col = client.get_collection(c.name)
        dump_collection(col)


if __name__ == "__main__":
    main()
