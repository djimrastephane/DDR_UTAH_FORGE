from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    _NX_OK = True
except ImportError:
    _NX_OK = False
    logger.warning("networkx not installed — graph building unavailable")

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ddr_rag.vocab import (
    tokenize_for_graph,
    label_op_code,
    label_phase,
    PHASE_LABELS,
    OP_CODE_LABELS,
)

DEFAULT_MIN_EDGE_WEIGHT   = 3
DEFAULT_MIN_NODE_DEGREE   = 2
DEFAULT_WINDOW_SIZE       = 0
DEFAULT_MAX_NODES         = 120
DEFAULT_INCLUDE_OP_CODES  = True


def _load_corpus(processed_dir: Path) -> "pd.DataFrame":
    if not _PANDAS_OK:
        raise RuntimeError("pandas required")
    import pandas as pd

    frames = []
    for doc_dir in sorted(processed_dir.iterdir()):
        f = doc_dir / "ddr_facts.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    if not frames:
        raise FileNotFoundError(f"No ddr_facts.parquet found under {processed_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["doc_id", "report_date", "start_time"]).reset_index(drop=True)
    return df


def _row_tokens(row: "pd.Series", include_op_code: bool) -> list[str]:
    tokens = tokenize_for_graph(str(row.get("operation_text") or ""))
    if include_op_code:
        op = str(row.get("op_code") or "").strip()
        if op:
            tokens.append(f"op:{op}")
    return tokens


def _cooccurrence_counts(
    rows: list[list[str]],
    window_size: int,
) -> Counter:
    counts: Counter = Counter()
    n = len(rows)
    for i, tokens_i in enumerate(rows):
        window_tokens: list[str] = list(tokens_i)
        for j in range(1, window_size + 1):
            if i + j < n:
                window_tokens.extend(rows[i + j])
        unique = sorted(set(window_tokens))
        for a, b in combinations(unique, 2):
            counts[(a, b)] += 1
    return counts


def _node_npt_ratio(
    token: str,
    token_rows: dict[str, list[bool]],
) -> float:
    flags = token_rows.get(token, [])
    if not flags:
        return 0.0
    return sum(flags) / len(flags)


def build_phase_graph(
    ops_df: "pd.DataFrame",
    phase: str,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_edge_weight: int = DEFAULT_MIN_EDGE_WEIGHT,
    min_node_degree: int = DEFAULT_MIN_NODE_DEGREE,
    include_op_code_nodes: bool = DEFAULT_INCLUDE_OP_CODES,
    max_nodes: int | None = DEFAULT_MAX_NODES,
) -> "nx.Graph":
    if not _NX_OK:
        raise RuntimeError("networkx required")

    phase_df = ops_df if phase == "ALL" else ops_df[ops_df["phase"] == phase].copy()
    if len(phase_df) == 0:
        logger.warning("No rows for phase %s", phase)
        return nx.Graph()

    rows_tokens: list[list[str]] = []
    token_freq: Counter = Counter()
    token_npt: dict[str, list[bool]] = defaultdict(list)

    for _, row in phase_df.iterrows():
        tokens = _row_tokens(row, include_op_code_nodes)
        if not tokens:
            continue
        rows_tokens.append(tokens)
        is_npt = bool(row.get("is_npt", False))
        unique_tok = set(tokens)
        for tok in unique_tok:
            token_freq[tok] += 1
            token_npt[tok].append(is_npt)

    if not rows_tokens:
        return nx.Graph()

    edge_counts = _cooccurrence_counts(rows_tokens, window_size)

    G: nx.Graph = nx.Graph()
    G.graph["phase"] = phase
    G.graph["phase_label"] = label_phase(phase)
    G.graph["n_ops"] = len(phase_df)
    G.graph["n_docs"] = phase_df["doc_id"].nunique()
    G.graph["window_size"] = window_size
    G.graph["min_edge_weight"] = min_edge_weight

    for (a, b), weight in edge_counts.items():
        if weight >= min_edge_weight:
            G.add_edge(a, b, weight=weight)

    for node in list(G.nodes()):
        freq = token_freq.get(node, 0)
        npt_ratio = _node_npt_ratio(node, token_npt)
        if node.startswith("op:"):
            raw_code = node[3:]
            node_type = "op_code"
            node_label = label_op_code(raw_code)
        else:
            node_type = "term"
            node_label = node
        G.nodes[node]["label"] = node_label
        G.nodes[node]["node_type"] = node_type
        G.nodes[node]["count"] = freq
        G.nodes[node]["npt_ratio"] = round(npt_ratio, 3)

    low_degree = [n for n, d in G.degree() if d < min_node_degree]
    G.remove_nodes_from(low_degree)

    G.remove_nodes_from(list(nx.isolates(G)))

    # Cap to max_nodes by weighted degree (sum of edge weights)
    if max_nodes is not None and G.number_of_nodes() > max_nodes:
        weighted_deg = {
            n: sum(d.get("weight", 1) for _, d in G[n].items())
            for n in G.nodes()
        }
        # Always keep op_code nodes; rank the rest by weighted degree
        op_nodes = [n for n in G.nodes() if G.nodes[n]["node_type"] == "op_code"]
        term_nodes = sorted(
            [n for n in G.nodes() if G.nodes[n]["node_type"] == "term"],
            key=lambda n: weighted_deg.get(n, 0),
            reverse=True,
        )
        keep = set(op_nodes) | set(term_nodes[: max(0, max_nodes - len(op_nodes))])
        G.remove_nodes_from([n for n in list(G.nodes()) if n not in keep])
        G.remove_nodes_from(list(nx.isolates(G)))

    if G.number_of_edges() > 0:
        max_w = max(d.get("weight", 1) for _, _, d in G.edges(data=True))
        for u, v in G.edges():
            G[u][v]["norm_weight"] = round(G[u][v].get("weight", 1) / max_w, 4)

    # Community detection via greedy modularity (works on disconnected graphs)
    if G.number_of_nodes() > 0:
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
            for cid, members in enumerate(communities):
                for node in members:
                    if node in G.nodes:
                        G.nodes[node]["community"] = cid
        except Exception as exc:
            logger.debug("Community detection failed: %s", exc)
            for node in G.nodes():
                G.nodes[node]["community"] = 0

    for node in G.nodes():
        G.nodes[node]["degree"] = G.degree(node)

    # Betweenness centrality (skip for very large graphs)
    if 0 < G.number_of_nodes() <= 300:
        try:
            bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
            for node, score in bc.items():
                G.nodes[node]["betweenness"] = round(score, 4)
        except Exception:
            pass

    return G


def build_all_graphs(
    processed_dir: Path,
    **kwargs: Any,
) -> "dict[str, nx.Graph]":
    ops_df = _load_corpus(processed_dir)
    graphs: dict[str, nx.Graph] = {}
    phases = sorted(ops_df["phase"].dropna().unique())
    logger.info("Building graphs for phases: %s", phases)
    for phase in phases:
        G = build_phase_graph(ops_df, phase, **kwargs)
        graphs[phase] = G
        logger.info(
            "  %s: %d nodes, %d edges",
            phase, G.number_of_nodes(), G.number_of_edges(),
        )
    return graphs


def graph_to_json(
    G: "nx.Graph",
    phase: str,
    ops_df: "pd.DataFrame",
) -> dict:
    phase_df = ops_df if phase == "ALL" else ops_df[ops_df["phase"] == phase]

    total_h = float(phase_df["duration_hr"].sum() or 0)
    npt_h   = float(phase_df.loc[phase_df["is_npt"], "duration_hr"].sum() or 0)

    nodes = []
    for nid, attrs in G.nodes(data=True):
        nodes.append({
            "id":          nid,
            "label":       attrs.get("label", nid),
            "node_type":   attrs.get("node_type", "term"),
            "count":       attrs.get("count", 0),
            "degree":      attrs.get("degree", 0),
            "npt_ratio":   attrs.get("npt_ratio", 0.0),
            "community":   attrs.get("community", 0),
            "betweenness": attrs.get("betweenness", 0.0),
        })

    links = []
    for u, v, attrs in G.edges(data=True):
        links.append({
            "source":      u,
            "target":      v,
            "weight":      attrs.get("weight", 1),
            "norm_weight": attrs.get("norm_weight", 1.0),
        })

    return {
        "phase":           phase,
        "phase_label":     label_phase(phase),
        "n_ops":           int(len(phase_df)),
        "n_docs":          int(phase_df["doc_id"].nunique()),
        "total_hours":     round(total_h, 1),
        "npt_hours":       round(npt_h, 1),
        "npt_pct":         round(100 * npt_h / total_h, 1) if total_h else 0.0,
        "n_nodes":         G.number_of_nodes(),
        "n_edges":         G.number_of_edges(),
        "n_communities":   len({d.get("community", 0) for _, d in G.nodes(data=True)}),
        "window_size":     G.graph.get("window_size", 0),
        "min_edge_weight": G.graph.get("min_edge_weight", DEFAULT_MIN_EDGE_WEIGHT),
        "nodes":           sorted(nodes, key=lambda n: -n["degree"]),
        "links":           sorted(links, key=lambda e: -e["weight"]),
    }


def save_graph(
    G: "nx.Graph",
    phase: str,
    ops_df: "pd.DataFrame",
    out_dir: Path,
) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)

    payload = graph_to_json(G, phase, ops_df)
    (out_dir / "graph.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    nx.write_graphml(G, str(out_dir / "graph.graphml"))

    node_records = []
    for nid, attrs in G.nodes(data=True):
        node_records.append({"node_id": nid, **attrs})
    pd.DataFrame(node_records).to_parquet(out_dir / "nodes.parquet", index=False)

    edge_records = []
    for u, v, attrs in G.edges(data=True):
        edge_records.append({"source": u, "target": v, **attrs})
    pd.DataFrame(edge_records).to_parquet(out_dir / "edges.parquet", index=False)


def build_and_save_all(
    processed_dir: Path,
    out_dir: Path,
    **kwargs: Any,
) -> "dict[str, nx.Graph]":
    ops_df = _load_corpus(processed_dir)
    graphs = build_all_graphs(processed_dir, **kwargs)

    summary: list[dict] = []
    for phase, G in graphs.items():
        phase_out = out_dir / phase
        save_graph(G, phase, ops_df, phase_out)
        summary.append({
            "phase":         phase,
            "phase_label":   label_phase(phase),
            "n_ops":         int(len(ops_df[ops_df["phase"] == phase])),
            "n_nodes":       G.number_of_nodes(),
            "n_edges":       G.number_of_edges(),
            "n_communities": len({d.get("community", 0) for _, d in G.nodes(data=True)}),
            "top_nodes":     [
                n for n, _ in sorted(
                    G.degree(), key=lambda x: -x[1]
                )[:8]
            ],
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return graphs
