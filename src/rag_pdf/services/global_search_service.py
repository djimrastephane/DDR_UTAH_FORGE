"""
Global cross-document search service for the DDR corpus.

Loads the pre-built global FAISS index (data/global_index/) and supports
hybrid BM25 + dense retrieval across all 14,879 chunks from all 171 DDRs.

Usage
-----
    from rag_pdf.services.global_search_service import GlobalSearchService

    svc = GlobalSearchService(repo_root=Path("."))
    results = svc.search("What was the highest NPT phase on this well?", k=10)
    for r in results:
        print(r["doc_id"], r["report_date"], r["score"], r["chunk_text"][:120])
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from rag_pdf.retrieval.hybrid_utils import BM25Index, tokenize

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

logger = logging.getLogger(__name__)

_GLOBAL_INDEX_DIR = "data/global_index"
_DEFAULT_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
_BM25_WEIGHT      = 0.3
_DENSE_WEIGHT     = 0.7


def _resolve_model_path(repo_root: Path) -> Path:
    local = repo_root / "models" / "all-MiniLM-L6-v2"
    return local if local.exists() else Path(_DEFAULT_MODEL)


class GlobalSearchService:
    """
    Hybrid BM25 + dense search across the full DDR corpus.

    Loads once; search() is stateless and thread-safe after init.
    """

    def __init__(
        self,
        repo_root: Path,
        index_dir: Path | None = None,
        model_path: Path | None = None,
    ) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self._repo_root = Path(repo_root)
        idx_dir = index_dir or (self._repo_root / _GLOBAL_INDEX_DIR)

        if not idx_dir.exists():
            raise FileNotFoundError(
                f"Global index not found at {idx_dir}. "
                "Run: python scripts/build_global_index.py"
            )

        logger.info("Loading global FAISS index from %s", idx_dir)
        self._index = faiss.read_index(str(idx_dir / "faiss.index"))
        self._meta  = pd.read_parquet(idx_dir / "chunk_meta.parquet")
        self._chunks = pd.read_parquet(idx_dir / "chunks.parquet")

        # Align meta and chunks on chunk_id_global
        if "chunk_id_global" in self._chunks.columns:
            self._chunks = self._chunks.set_index("chunk_id_global")

        # BM25 index over all chunk texts
        logger.info("Building BM25 index over %d chunks...", len(self._meta))
        texts = self._chunks.reindex(
            self._meta["chunk_id_global"]
        )["chunk_text"].fillna("").tolist()
        tokenised = [tokenize(t) for t in texts]
        self._bm25 = BM25Index(tokenised)
        self._texts = texts   # keep for snippet extraction

        # Embedding model
        mp = model_path or _resolve_model_path(self._repo_root)
        logger.info("Loading embedding model: %s", mp)
        self._model = SentenceTransformer(str(mp), device="cpu")

        logger.info("GlobalSearchService ready (%d chunks)", len(self._meta))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        question: str,
        k: int = 10,
        doc_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict]:
        """
        Hybrid search across the full corpus.

        Parameters
        ----------
        question:    Natural language query.
        k:           Number of results to return.
        doc_filter:  Optional doc_id prefix filter (e.g. "RigAlpha-DDR-085").
        date_from:   ISO date string "YYYY-MM-DD" — filter results on or after.
        date_to:     ISO date string "YYYY-MM-DD" — filter results on or before.

        Returns
        -------
        List of result dicts, sorted by combined score descending.
        """
        # --- Dense retrieval ---
        q_emb = self._model.encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

        fetch_k = min(k * 8, len(self._meta))
        dense_scores, dense_indices = self._index.search(q_emb, fetch_k)
        dense_scores  = dense_scores[0].tolist()
        dense_indices = dense_indices[0].tolist()

        # --- BM25 retrieval ---
        tokens   = tokenize(question)
        bm25_raw = np.asarray(self._bm25.score_query(tokens), dtype=np.float32)
        bm25_top_idx = np.argsort(bm25_raw)[::-1][:fetch_k].tolist()
        bm25_top_scores = bm25_raw[bm25_top_idx].tolist()

        # Normalise both score lists to [0, 1]
        def _norm(scores: list[float]) -> list[float]:
            mx = max(scores) if scores else 1.0
            mn = min(scores) if scores else 0.0
            rng = mx - mn or 1.0
            return [(s - mn) / rng for s in scores]

        d_norm = _norm(dense_scores)
        b_norm = _norm(bm25_top_scores)

        # Combine into score map {row_index: combined_score}
        score_map: dict[int, float] = {}
        for idx, sc in zip(dense_indices, d_norm):
            if idx < 0:
                continue
            score_map[idx] = score_map.get(idx, 0.0) + _DENSE_WEIGHT * sc
        for idx, sc in zip(bm25_top_idx, b_norm):
            score_map[idx] = score_map.get(idx, 0.0) + _BM25_WEIGHT * sc

        # Sort by combined score
        ranked = sorted(score_map.items(), key=lambda x: -x[1])

        # Build results
        results = []
        for row_idx, score in ranked:
            if len(results) >= k:
                break
            if row_idx >= len(self._meta):
                continue
            meta_row = self._meta.iloc[row_idx]
            doc_id   = str(meta_row.get("doc_id", ""))
            cid_g    = str(meta_row.get("chunk_id_global", ""))

            # Filters
            if doc_filter and not doc_id.startswith(doc_filter):
                continue
            report_date = str(meta_row.get("report_date", ""))
            if date_from and report_date and report_date < date_from:
                continue
            if date_to and report_date and report_date > date_to:
                continue

            # Chunk text
            if cid_g in self._chunks.index:
                cr = self._chunks.loc[cid_g]
                chunk_text = str(cr.get("chunk_text", ""))
                section    = str(cr.get("section_title", ""))
                is_table   = bool(cr.get("is_table", False))
            else:
                chunk_text = self._texts[row_idx] if row_idx < len(self._texts) else ""
                section    = str(meta_row.get("section_title", ""))
                is_table   = bool(meta_row.get("is_table", False))

            results.append({
                "rank":         len(results) + 1,
                "score":        round(score, 4),
                "doc_id":       doc_id,
                "report_date":  report_date,
                "wellbore":     str(meta_row.get("wellbore", "")),
                "report_no":    str(meta_row.get("report_no", "")),
                "chunk_id_global": cid_g,
                "page_start":   meta_row.get("page_start"),
                "is_table":     is_table,
                "section_title": section,
                "chunk_text":   chunk_text,
                "snippet":      chunk_text[:300],
            })

        return results
