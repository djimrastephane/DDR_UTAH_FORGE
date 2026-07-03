from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

PROCESSED_DIR = repo_root / "data" / "processed"
OUT_DIR       = repo_root / "data" / "global_index"

META_COLS = [
    "doc_id", "chunk_id", "chunk_id_global",
    "section_title", "subsection_title", "is_table",
    "page_start", "page_end", "pages",
]

CHUNK_COLS = [
    "doc_id", "chunk_id", "chunk_id_global",
    "chunk_text", "is_table", "section_title",
    "page_start", "page_end",
]


def _parse_date(s: str) -> str:
    try:
        return datetime.strptime(s.strip(), "%d/%m/%Y").strftime("%Y-%m-%d")
    except Exception:
        return str(s or "")


def main() -> None:
    import faiss

    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = sorted([
        d for d in PROCESSED_DIR.iterdir()
        if d.is_dir()
        and (d / "embeddings.npy").exists()
        and (d / "chunk_meta.parquet").exists()
        and (d / "chunks.parquet").exists()
    ])
    print(f"Found {len(docs)} docs with embeddings")

    header_map: dict[str, dict] = {}
    for d in docs:
        hdr_path = d / "ddr_header.parquet"
        if hdr_path.exists():
            hdr = pd.read_parquet(hdr_path).iloc[0]
            header_map[d.name] = {
                "report_date": _parse_date(str(hdr.get("report_date", ""))),
                "wellbore":    str(hdr.get("wellbore", "")),
                "report_no":   str(hdr.get("report_no", "")),
            }
        else:
            header_map[d.name] = {"report_date": "", "wellbore": "", "report_no": ""}

    all_embs:   list[np.ndarray]    = []
    all_meta:   list[pd.DataFrame]  = []
    all_chunks: list[pd.DataFrame]  = []
    total = 0

    for i, doc_dir in enumerate(docs, 1):
        doc_id = doc_dir.name
        emb  = np.load(doc_dir / "embeddings.npy").astype("float32")
        meta = pd.read_parquet(doc_dir / "chunk_meta.parquet")
        cks  = pd.read_parquet(doc_dir / "chunks.parquet")

        if emb.shape[0] != len(meta):
            print(f"  SKIP {doc_id}: embedding rows {emb.shape[0]} ≠ meta rows {len(meta)}")
            continue

        h = header_map.get(doc_id, {})
        meta = meta.copy()
        meta["report_date"] = h.get("report_date", "")
        meta["wellbore"]    = h.get("wellbore", "")
        meta["report_no"]   = h.get("report_no", "")

        cks = cks.copy()
        cks["report_date"] = h.get("report_date", "")
        cks["wellbore"]    = h.get("wellbore", "")

        all_embs.append(emb)
        meta_keep = [c for c in META_COLS + ["report_date", "wellbore", "report_no"]
                     if c in meta.columns]
        all_meta.append(meta[meta_keep])
        chunk_keep = [c for c in CHUNK_COLS + ["report_date", "wellbore"]
                      if c in cks.columns]
        all_chunks.append(cks[chunk_keep])

        total += emb.shape[0]
        if i % 20 == 0:
            print(f"  {i}/{len(docs)}  {total:,} chunks so far")

    print(f"Concatenating {total:,} chunks from {len(all_embs)} docs...")

    embeddings = np.vstack(all_embs).astype("float32")
    meta_df    = pd.concat(all_meta,   ignore_index=True)
    chunks_df  = pd.concat(all_chunks, ignore_index=True)

    assert len(embeddings) == len(meta_df), \
        f"Alignment error: {len(embeddings)} emb rows vs {len(meta_df)} meta rows"

    print(f"Building FAISS IndexFlatIP ({embeddings.shape})...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(OUT_DIR / "faiss.index"))
    np.save(OUT_DIR / "embeddings.npy", embeddings)
    meta_df.to_parquet(OUT_DIR / "chunk_meta.parquet", index=False)
    chunks_df.to_parquet(OUT_DIR / "chunks.parquet", index=False)

    elapsed = round(time.perf_counter() - t0, 2)
    metrics = {
        "built_utc":     datetime.now(timezone.utc).isoformat(),
        "n_docs":        len(all_embs),
        "n_chunks":      int(total),
        "embedding_dim": int(embeddings.shape[1]),
        "faiss_type":    "IndexFlatIP",
        "normalised":    True,
        "elapsed_s":     elapsed,
        "meta_columns":  list(meta_df.columns),
        "chunk_columns": list(chunks_df.columns),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nGlobal index written to {OUT_DIR}/")
    print(f"  faiss.index     — {total:,} vectors × {embeddings.shape[1]}d")
    print(f"  chunk_meta      — {len(meta_df):,} rows, cols: {list(meta_df.columns)}")
    print(f"  chunks          — {len(chunks_df):,} rows")
    print(f"  elapsed: {elapsed}s")


if __name__ == "__main__":
    main()
