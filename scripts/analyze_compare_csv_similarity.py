"""speaker_verify_compare.csv 에서 baseline / finetuned 유사도 평균."""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "app" / "logs" / "speaker_verify_compare.csv",
        root / "logs" / "speaker_verify_compare.csv",
    ]
    if len(sys.argv) > 1:
        candidates.insert(0, Path(sys.argv[1]))
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        print("CSV 없음. 경로:", ", ".join(str(p) for p in candidates))
        sys.exit(1)

    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    bl: list[float] = []
    ft: list[float] = []
    for r in rows:
        try:
            b = float(r.get("baseline_similarity") or "")
            f = float(r.get("finetuned_similarity") or "")
        except ValueError:
            continue
        if (r.get("bypass") or "").lower() == "true":
            continue
        if b < 0 or f < 0:
            continue
        bl.append(b)
        ft.append(f)

    n = len(bl)
    print("file:", path.resolve())
    print("rows_total:", len(rows))
    print("rows_used (bypass=false, sim>=0):", n)
    if n == 0:
        sys.exit(0)
    print("baseline_mean:", sum(bl) / n)
    print("finetuned_mean:", sum(ft) / n)
    cids = {r.get("call_id", "") for r in rows if r.get("call_id")}
    print("distinct_call_id:", len(cids))


if __name__ == "__main__":
    main()
