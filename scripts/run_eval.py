from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from rag_pdf.services.search_service import SearchService

PROCESSED_DIR = repo_root / "data" / "processed"
K_VALUES = [1, 3, 5, 10]


def _model_path() -> Path:
    local = repo_root / "models" / "all-MiniLM-L6-v2"
    return local if local.exists() else Path("sentence-transformers/all-MiniLM-L6-v2")


def _keyword_hit(chunks: list[dict], keyword: str) -> tuple[bool, int]:
    if not keyword:
        return False, 0
    kw_lower = keyword.lower()
    for rank, chunk in enumerate(chunks, 1):
        text = str(chunk.get("chunk_text", "")).lower()
        if kw_lower in text:
            return True, rank
    return False, 0


def _page_hit(chunks: list[dict], expected_pages: list[int] | None) -> tuple[bool, int]:
    if not expected_pages:
        return False, 0
    expected_set = set(expected_pages)
    for rank, chunk in enumerate(chunks, 1):
        # Primary: 'pages' list (SearchService format)
        pages_val = chunk.get("pages")
        if isinstance(pages_val, list):
            if any(int(p) in expected_set for p in pages_val if p is not None):
                return True, rank
        # Fallback: page_start / page_end (chunk_meta format)
        for key in ("page_start", "page_end"):
            v = chunk.get(key)
            if v is not None:
                try:
                    if int(v) in expected_set:
                        return True, rank
                except (TypeError, ValueError):
                    pass
    return False, 0


def _recall_at_k(results: list[dict], k: int, signal: str) -> float:
    evaluable = [r for r in results if r["evaluable"] and r[f"{signal}_rank"] > 0 or r[f"{signal}_rank"] == 0]
    # Only count questions where the signal is applicable
    applicable = [r for r in results if r["evaluable"] and r.get(f"has_{signal}")]
    if not applicable:
        return float("nan")
    hits = sum(1 for r in applicable if 0 < r[f"{signal}_rank"] <= k)
    return hits / len(applicable)


def run_eval(
    eval_path: Path,
    max_k: int = 10,
) -> list[dict]:
    payload = json.loads(eval_path.read_text())
    questions = payload["questions"]
    evaluable_qs = [q for q in questions if q.get("doc_id")]
    manual_qs   = [q for q in questions if not q.get("doc_id")]

    print(f"Loaded {len(questions)} questions: {len(evaluable_qs)} retrievable, {len(manual_qs)} cross-doc")
    print(f"Loading per-doc search model...")
    service = SearchService(repo_root=repo_root, model_path=_model_path())
    print("Per-doc model loaded.\n")

    # Load global search service if global index exists
    global_svc = None
    global_index_dir = repo_root / "data" / "global_index"
    if global_index_dir.exists():
        try:
            from rag_pdf.services.global_search_service import GlobalSearchService
            print("Loading global search service...")
            global_svc = GlobalSearchService(repo_root=repo_root)
            print("Global service loaded.\n")
        except Exception as exc:
            print(f"Global service unavailable: {exc}\n")

    results: list[dict] = []

    for i, q in enumerate(evaluable_qs, 1):
        qid     = q["query_id"]
        doc_id  = q["doc_id"]
        question = q["question"]
        keyword  = q.get("evidence_keyword") or ""
        exp_pages = q.get("expected_pages") or []

        data_dir = PROCESSED_DIR / doc_id
        if not data_dir.exists():
            print(f"  [{qid}] SKIP — doc dir missing: {doc_id}")
            continue

        t0 = time.perf_counter()
        try:
            out = service.search(
                data_dir=data_dir,
                question=question,
                k=max_k,
                include_generated_answer=False,
            )
            elapsed = time.perf_counter() - t0
            chunks = out.get("results", [])
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  [{qid}] ERROR: {exc}")
            results.append({
                "query_id": qid, "doc_id": doc_id, "evaluable": True,
                "error": str(exc), "has_keyword": bool(keyword),
                "has_page": bool(exp_pages),
                "keyword_rank": 0, "page_rank": 0,
                "hit_rank": 0, "top_score": 0.0, "elapsed_s": round(elapsed, 3),
                "category": q["category"], "difficulty": q["difficulty"],
            })
            continue

        kw_hit, kw_rank   = _keyword_hit(chunks, keyword)
        pg_hit, pg_rank   = _page_hit(chunks, exp_pages)

        # Primary hit: keyword (preferred) OR page if keyword absent
        if keyword:
            hit_rank = kw_rank
        elif exp_pages:
            hit_rank = pg_rank
        else:
            hit_rank = 0

        top_score = float(chunks[0].get("score", 0)) if chunks else 0.0

        status = (
            f"✓ rank={hit_rank}" if hit_rank else
            f"✗ (kw={'found@'+str(kw_rank) if kw_hit else 'miss'}, "
            f"pg={'found@'+str(pg_rank) if pg_hit else 'miss'})"
        )
        print(f"  [{qid}] {q['category']}/{q['difficulty']:6} {status:30}  {top_score:.3f}  {elapsed:.2f}s  {question[:60]}")

        results.append({
            "query_id":    qid,
            "doc_id":      doc_id,
            "evaluable":   True,
            "error":       "",
            "category":    q["category"],
            "difficulty":  q["difficulty"],
            "has_keyword": bool(keyword),
            "has_page":    bool(exp_pages),
            "keyword_rank": kw_rank,
            "page_rank":   pg_rank,
            "hit_rank":    hit_rank,
            "top_score":   top_score,
            "elapsed_s":   round(elapsed, 3),
            "question":    question,
            "expected_answer": q.get("expected_answer", ""),
            "evidence_keyword": keyword,
            "expected_pages": exp_pages,
            "top_chunks": [
                {
                    "rank":       r + 1,
                    "score":      float(chunks[r].get("score", 0)),
                    "page_start": chunks[r].get("page_start"),
                    "chunk_text": str(chunks[r].get("chunk_text", ""))[:300],
                }
                for r in range(min(5, len(chunks)))
            ],
        })

    # These cannot be answered by text chunk retrieval — they require
    # computing over ddr_facts.parquet / ddr_header.parquet directly.
    def _structured_phase_npt() -> str:
        frames = [pd.read_parquet(d / "ddr_facts.parquet")
                  for d in PROCESSED_DIR.iterdir()
                  if (d / "ddr_facts.parquet").exists()]
        df = pd.concat(frames, ignore_index=True)
        stats = df.groupby("phase").apply(
            lambda g: pd.Series({
                "npt_hrs": g.loc[g["is_npt"], "duration_hr"].sum(),
                "npt_pct": 100 * g.loc[g["is_npt"], "duration_hr"].sum()
                           / max(g["duration_hr"].sum(), 1),
            })
        ).reset_index()
        # Sort by npt_hrs (most operationally significant) not by npt_pct
        top = stats.sort_values("npt_hrs", ascending=False).iloc[0]
        return f"PROD1 ({top['npt_pct']:.0f}% NPT, {top['npt_hrs']:.0f}h)"

    def _structured_max_depth() -> str:
        import re
        def _num(s):
            try: return float(re.sub(r"[^\d.]", "", str(s).split()[0]))
            except: return 0.0
        frames = [pd.read_parquet(d / "ddr_header.parquet")
                  for d in PROCESSED_DIR.iterdir()
                  if (d / "ddr_header.parquet").exists()]
        hdr = pd.concat(frames, ignore_index=True)
        hdr["d"] = hdr["end_depth_md_ft"].apply(_num)
        return f"{hdr['d'].max():,.0f} ft MD"

    STRUCTURED_ANSWERS = {
        "X01": (_structured_phase_npt,  "PROD1"),
        "X02": (_structured_max_depth,  "19,127"),
    }

    print("\nCross-document questions:")
    for q in manual_qs:
        qid      = q["query_id"]
        question = q["question"]
        keyword  = q.get("evidence_keyword") or ""

        # Structured aggregate questions bypass text retrieval
        if qid in STRUCTURED_ANSWERS:
            fn, expect_kw = STRUCTURED_ANSWERS[qid]
            t0 = time.perf_counter()
            try:
                answer = fn()
                elapsed = time.perf_counter() - t0
                hit = expect_kw.lower() in answer.lower()
                print(f"  [{qid}] structured query → {answer}  {'✓ HIT' if hit else '✗ MISS'} ({elapsed:.2f}s)")
                results.append({
                    "query_id": qid, "doc_id": None, "evaluable": True,
                    "error": "", "category": q["category"], "difficulty": q["difficulty"],
                    "has_keyword": True, "has_page": False,
                    "keyword_rank": 1 if hit else 0,
                    "page_rank": 0, "hit_rank": 1 if hit else 0,
                    "top_score": 1.0 if hit else 0.0, "elapsed_s": round(elapsed, 3),
                    "question": question, "expected_answer": q.get("expected_answer", ""),
                    "note": f"structured_data_query: {answer}",
                })
            except Exception as exc:
                print(f"  [{qid}] structured query FAILED: {exc}")
                results.append({
                    "query_id": qid, "doc_id": None, "evaluable": False,
                    "error": str(exc), "category": q["category"], "difficulty": q["difficulty"],
                    "question": question, "note": "structured query failed",
                })
            continue

        if global_svc is None:
            results.append({
                "query_id": qid, "doc_id": None, "evaluable": False,
                "category": q["category"], "difficulty": q["difficulty"],
                "question": question, "expected_answer": q.get("expected_answer", ""),
                "note": "Global index not found",
            })
            continue

        t0 = time.perf_counter()
        try:
            chunks = global_svc.search(question, k=max_k)
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  [{qid}] ERROR: {exc}")
            results.append({
                "query_id": qid, "doc_id": None, "evaluable": True,
                "error": str(exc), "category": q["category"],
                "difficulty": q["difficulty"], "question": question,
                "has_keyword": bool(keyword), "has_page": False,
                "keyword_rank": 0, "page_rank": 0, "hit_rank": 0,
                "top_score": 0.0, "elapsed_s": round(elapsed, 3),
            })
            continue

        kw_hit, kw_rank = _keyword_hit(chunks, keyword)
        hit_rank = kw_rank if keyword else 0

        top_docs = list({r["doc_id"] for r in chunks[:5]})
        top_score = float(chunks[0].get("score", 0)) if chunks else 0.0

        status = f"✓ rank={hit_rank}" if hit_rank else (
            f"~ top_docs={[d[-15:] for d in top_docs[:3]]}" if chunks else "✗ no results"
        )
        print(f"  [{qid}] {q['category']}/{q['difficulty']:6} {status:50}  {top_score:.3f}  {elapsed:.2f}s")

        results.append({
            "query_id":    qid,
            "doc_id":      None,
            "evaluable":   True,
            "error":       "",
            "category":    q["category"],
            "difficulty":  q["difficulty"],
            "has_keyword": bool(keyword),
            "has_page":    False,
            "keyword_rank": kw_rank,
            "page_rank":   0,
            "hit_rank":    hit_rank,
            "top_score":   top_score,
            "elapsed_s":   round(elapsed, 3),
            "question":    question,
            "expected_answer": q.get("expected_answer", ""),
            "evidence_keyword": keyword,
            "top_docs":    top_docs[:5],
            "top_chunks": [
                {
                    "rank":      r + 1,
                    "score":     float(chunks[r].get("score", 0)),
                    "doc_id":    chunks[r].get("doc_id", ""),
                    "report_date": chunks[r].get("report_date", ""),
                    "chunk_text": str(chunks[r].get("chunk_text", ""))[:300],
                }
                for r in range(min(5, len(chunks)))
            ],
        })

    return results


def _fmt(v: float) -> str:
    return f"{v:.1%}" if not (v != v) else "n/a"


def print_summary(results: list[dict]) -> None:
    evaluable = [r for r in results if r.get("evaluable") and not r.get("error")]

    print()
    print("=" * 65)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 65)

    print(f"\nOverall (keyword-based, {len(evaluable)} questions evaluated)")
    print(f"{'Metric':<18}", end="")
    for k in K_VALUES:
        print(f"  @{k:2}", end="")
    print()
    print("-" * 42)

    for signal, label in [("keyword", "Keyword hit"), ("page", "Page hit"), ("hit", "Primary hit")]:
        applicable = [r for r in evaluable if r.get(f"has_{signal}") or signal == "hit"]
        if signal == "hit":
            applicable = [r for r in evaluable if r.get("has_keyword") or r.get("has_page")]
        print(f"  {label:<16}", end="")
        for k in K_VALUES:
            hits = sum(1 for r in applicable if 0 < r.get(f"{signal}_rank", 0) <= k)
            pct = hits / len(applicable) if applicable else float("nan")
            print(f"  {_fmt(pct):>4}", end="")
        print(f"  (n={len(applicable)})")

    print("\nRecall@5 by category (primary hit)")
    categories = sorted({r["category"] for r in evaluable})
    for cat in categories:
        cat_qs = [r for r in evaluable if r["category"] == cat
                  and (r.get("has_keyword") or r.get("has_page"))]
        if not cat_qs:
            continue
        hits5 = sum(1 for r in cat_qs if 0 < r.get("hit_rank", 0) <= 5)
        hits1 = sum(1 for r in cat_qs if 0 < r.get("hit_rank", 0) <= 1)
        print(f"  {cat:<25}  @1={_fmt(hits1/len(cat_qs))}  @5={_fmt(hits5/len(cat_qs))}  (n={len(cat_qs)})")

    print("\nRecall@5 by difficulty (primary hit)")
    for diff in ["easy", "medium", "hard"]:
        diff_qs = [r for r in evaluable if r["difficulty"] == diff
                   and (r.get("has_keyword") or r.get("has_page"))]
        if not diff_qs:
            continue
        hits5 = sum(1 for r in diff_qs if 0 < r.get("hit_rank", 0) <= 5)
        print(f"  {diff:<10}  @5={_fmt(hits5/len(diff_qs))}  (n={len(diff_qs)})")

    misses = [r for r in evaluable
              if r.get("hit_rank", 0) == 0
              and (r.get("has_keyword") or r.get("has_page"))]
    if misses:
        print(f"\nMissed questions (hit_rank=0, n={len(misses)}):")
        for r in misses:
            print(f"  [{r['query_id']}] {r['category']}/{r['difficulty']}  kw={r.get('evidence_keyword')!r}")
            print(f"       Q: {r['question'][:80]}")
            if r.get("top_chunks"):
                top = r["top_chunks"][0]
                print(f"       Top1 score={top['score']:.3f}: {top['chunk_text'][:80]}")

    cross_doc = [r for r in results if r.get("category") == "cross_document"]
    if cross_doc:
        evaluated = [r for r in cross_doc if r.get("evaluable") and not r.get("error")]
        manual    = [r for r in cross_doc if not r.get("evaluable")]
        print(f"\nCross-document questions ({len(cross_doc)} total, {len(evaluated)} evaluated, {len(manual)} manual):")
        for r in cross_doc:
            if r.get("evaluable") and r.get("top_docs"):
                docs_str = ", ".join(d[-20:] for d in (r.get("top_docs") or [])[:3])
                print(f"  [{r['query_id']}] top docs: {docs_str}")
            elif not r.get("evaluable"):
                print(f"  [{r['query_id']}] manual — {r['question'][:70]}")

    avg_elapsed = sum(r.get("elapsed_s", 0) for r in evaluable) / max(len(evaluable), 1)
    print(f"\nAvg retrieval time: {avg_elapsed:.2f}s/query")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(repo_root / "data/processed/qc/eval_questions.json"))
    parser.add_argument("--out",  default=str(repo_root / "data/processed/qc/eval_results.json"))
    parser.add_argument("--k",    type=int, default=10)
    args = parser.parse_args()

    print(f"DDR RAG Evaluation")
    print(f"  eval set : {args.eval}")
    print(f"  max k    : {args.k}")
    print()

    results = run_eval(Path(args.eval), max_k=args.k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False))

    print_summary(results)
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
