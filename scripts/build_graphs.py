from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from ddr_rag.graph_builder import build_and_save_all, build_phase_graph, _load_corpus, save_graph
from ddr_rag.vocab import label_phase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DDR section-level co-occurrence graphs")
    parser.add_argument("--data-dir",   default=str(repo_root / "data" / "processed"))
    parser.add_argument("--out-dir",    default=str(repo_root / "data" / "graphs"))
    parser.add_argument("--phase",      default="",    help="Build only this phase (e.g. PROD1)")
    parser.add_argument("--window",     type=int, default=0,   help="Co-occurrence window in rows (0=same row)")
    parser.add_argument("--min-edge",   type=int, default=3,   help="Min edge co-occurrence count")
    parser.add_argument("--min-degree", type=int, default=2,   help="Min node degree to keep")
    parser.add_argument("--max-nodes",  type=int, default=120, help="Max nodes per graph (0=no cap)")
    parser.add_argument("--no-op-codes", action="store_true",  help="Exclude op_code anchor nodes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    max_nodes = args.max_nodes if args.max_nodes > 0 else None

    kwargs = dict(
        window_size=args.window,
        min_edge_weight=args.min_edge,
        min_node_degree=args.min_degree,
        max_nodes=max_nodes,
        include_op_code_nodes=not args.no_op_codes,
    )

    if args.phase:
        ops_df = _load_corpus(data_dir)
        phase = args.phase.upper()
        print(f"Building graph for phase {phase} ({label_phase(phase)})...")
        G = build_phase_graph(ops_df, phase, **kwargs)
        save_graph(G, phase, ops_df, out_dir / phase)
        print(f"  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")
        print(f"  saved to {out_dir / phase}")
    else:
        print(f"Building graphs for all phases...")
        print(f"  data-dir : {data_dir}")
        print(f"  out-dir  : {out_dir}")
        print(f"  window   : {args.window} rows")
        print(f"  min-edge : {args.min_edge}")
        print(f"  max-nodes: {max_nodes or 'no cap'}")
        print()

        graphs = build_and_save_all(data_dir, out_dir, **kwargs)

        print()
        print("=== Graph summary ===")
        for phase in sorted(graphs):
            G = graphs[phase]
            top5 = [n for n, _ in sorted(G.degree(), key=lambda x: -x[1])[:5]]
            n_communities = len({d.get("community", 0) for _, d in G.nodes(data=True)})
            print(f"  {phase:<8} ({label_phase(phase):<35}): "
                  f"{G.number_of_nodes():3d} nodes  {G.number_of_edges():4d} edges  "
                  f"{n_communities} communities")
            print(f"           top nodes: {top5}")

        print(f"\nAll graphs saved to {out_dir}/")


if __name__ == "__main__":
    main()
