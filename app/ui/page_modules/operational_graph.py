from __future__ import annotations

import collections
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .constants import PHASE_ORDER, PHASE_COLOURS, PROCESSED_DIR
    from .loaders import load_graph
    from .utils import _apply_chart_theme
except ImportError:
    from constants import PHASE_ORDER, PHASE_COLOURS, PROCESSED_DIR  # type: ignore[no-redef]
    from loaders import load_graph                                     # type: ignore[no-redef]
    from utils import _apply_chart_theme                               # type: ignore[no-redef]

_ui_root = Path(__file__).resolve().parents[3]
if str(_ui_root / "src") not in sys.path:
    sys.path.insert(0, str(_ui_root / "src"))

from ddr_rag.vocab import label_phase, label_op_code, tokenize_for_graph


_EXCERPT_CHARS:    int = 200
_HOVER_EXCERPTS:   int = 3
_EDGE_LABEL_TOP_N: int = 8

_COMMUNITY_COLOURS: dict[int, str] = {
    0: "#1976D2",
    1: "#388E3C",
    2: "#F57C00",
    3: "#7B1FA2",
    4: "#0097A7",
    5: "#C62828",
    6: "#5D4037",
    7: "#455A64",
}

_GRAPH_MODES: dict[str, tuple[str, object]] = {
    "all":        ("All terms", None),
    "risk":       ("High NPT association terms",
                   lambda nid, a: a.get("npt_ratio", 0) > 0.35),
    "workflow":   ("Workflow anchors",
                   lambda nid, a: a.get("node_type") == "op_code"),
    "pressure":   ("Pressure & hydraulics",
                   lambda nid, a: any(t in nid for t in
                       ["pressure", "pump", "choke", "sbp", "mpd", "flowrate",
                        "surface-back", "ecd", "mud-weight", "flow"])),
    "completion": ("Completion execution",
                   lambda nid, a: any(t in nid for t in
                       ["sleeve", "packer", "frac", "ncs", "rih", "compression",
                        "wash-pipe", "mule-shoe", "flow-in-area", "tubing"])),
}


@st.cache_data(show_spinner=False)
def build_node_enrichment(phase: str, window: str) -> dict[str, dict]:
    frames = []
    for doc_dir in sorted(PROCESSED_DIR.iterdir()):
        f = doc_dir / "ddr_facts.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    if not frames:
        return {}

    ops = pd.concat(frames, ignore_index=True)
    ops["report_date_parsed"] = pd.to_datetime(
        ops["report_date"], dayfirst=True, errors="coerce"
    )
    if phase != "ALL":
        ops = ops[ops["phase"] == phase]
    if ops.empty:
        return {}

    term_rows: dict[str, list[dict]] = collections.defaultdict(list)
    for _, row in ops.iterrows():
        tokens = tokenize_for_graph(str(row.get("operation_text") or ""))
        dt  = row.get("report_date_parsed")
        dur = row.get("duration_hr")
        rec = {
            "date":   dt.strftime("%d %b %Y") if pd.notna(dt) else "—",
            "month":  dt.strftime("%b %Y")    if pd.notna(dt) else "",
            "dur":    round(float(dur), 2) if pd.notna(dur) else 0,
            "is_npt": bool(row.get("is_npt", False)),
            "pt_x":   str(row.get("pt_x") or ""),
            "text":   str(row.get("operation_text") or "")[:_EXCERPT_CHARS],
        }
        for tok in set(tokens):
            term_rows[tok].append(rec)

    enrichment: dict[str, dict] = {}
    for term, rows in term_rows.items():
        if not rows:
            continue
        dates  = [r["date"]  for r in rows]
        months = [r["month"] for r in rows if r["month"]]
        first_date = dates[0]  if dates  else "—"
        last_date  = dates[-1] if dates  else "—"

        month_counts = collections.Counter(months)
        peak_month   = month_counts.most_common(1)[0][0] if month_counts else ""

        if first_date == last_date:
            temporal_note = f"Single day: {first_date}"
        elif peak_month:
            temporal_note = f"Active {first_date} → {last_date} · peak: {peak_month}"
        else:
            temporal_note = f"Active {first_date} → {last_date}"

        npt_rows  = [r for r in rows if r["is_npt"]]
        prod_rows = [r for r in rows if not r["is_npt"]]
        pool = sorted(npt_rows, key=lambda r: -r["dur"])[:_HOVER_EXCERPTS]
        if len(pool) < _HOVER_EXCERPTS:
            pool += sorted(prod_rows, key=lambda r: -r["dur"])[: _HOVER_EXCERPTS - len(pool)]

        enrichment[term] = {
            "temporal_note": temporal_note,
            "excerpts":  [
                f"[{r['date']} · {r['dur']:.1f}h · {r['pt_x']}] {r['text']}"
                for r in pool[:_HOVER_EXCERPTS]
            ],
            "n_ops":     len(rows),
            "n_days":    len({r["date"] for r in rows if r["date"] != "—"}),
            "npt_ops":   sum(1 for r in rows if r["is_npt"]),
            "total_hrs": round(sum(r["dur"] for r in rows), 1),
            "npt_hrs":   round(sum(r["dur"] for r in rows if r["is_npt"]), 1),
        }
    return enrichment


def _build_nx(graph_data: dict, min_norm_weight: float) -> "nx.Graph":
    import networkx as nx
    G: nx.Graph = nx.Graph()
    for n in graph_data["nodes"]:
        G.add_node(n["id"], **n)
    for e in graph_data["links"]:
        if e.get("norm_weight", 0) >= min_norm_weight:
            G.add_edge(e["source"], e["target"],
                       weight=e["weight"], norm_weight=e.get("norm_weight", 0))
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def _apply_mode_filter(G: "nx.Graph", mode_key: str) -> "nx.Graph":
    import networkx as nx
    _, seed_fn = _GRAPH_MODES[mode_key]
    if seed_fn is None:
        return G
    seeds = {n for n, a in G.nodes(data=True) if seed_fn(n, a)}
    if not seeds:
        return G
    keep = set(seeds)
    for s in seeds:
        keep.update(G.neighbors(s))
    G2 = G.subgraph(keep).copy()
    G2.remove_nodes_from(list(nx.isolates(G2)))
    return G2


def render_graph(
    graph_data: dict,
    min_norm_weight: float = 0.80,
    mode_key: str = "all",
    max_labels: int = 8,
    enrichment: dict | None = None,
) -> go.Figure:
    import networkx as nx

    G = _build_nx(graph_data, min_norm_weight)
    G = _apply_mode_filter(G, mode_key)

    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No nodes match this filter. Try a lower edge threshold or a different mode.",
            showarrow=False, font=dict(size=13, color="#555"),
            x=0.5, y=0.5, xref="paper", yref="paper",
        )
        fig.update_layout(paper_bgcolor="#FAFAFA", plot_bgcolor="#FAFAFA",
                          height=400, xaxis_visible=False, yaxis_visible=False)
        return fig

    pos       = nx.spring_layout(G, weight="weight", seed=42, k=1.5)
    degrees   = dict(G.degree())
    n_max_deg = max(degrees.values(), default=1)
    label_set = {n for n, _ in sorted(degrees.items(), key=lambda x: -x[1])[:max_labels]}

    traces: list[go.BaseTraceType] = []

    for lo, hi, lw, col in [
        (0.70, 1.01, 3.0, "rgba(50,50,50,0.70)"),
        (0.40, 0.70, 1.6, "rgba(120,120,120,0.55)"),
        (0.00, 0.40, 0.7, "rgba(190,190,190,0.40)"),
    ]:
        ex, ey, etxt = [], [], []
        for u, v, data in G.edges(data=True):
            if lo <= data.get("norm_weight", 0) < hi:
                x0, y0 = pos[u]; x1, y1 = pos[v]
                ex += [x0, x1, None]; ey += [y0, y1, None]
                label_txt = f"{data.get('weight', '')} co-occurrences<br>{u} ↔ {v}"
                etxt += [label_txt, label_txt, None]
        if ex:
            traces.append(go.Scatter(
                x=ex, y=ey, mode="lines",
                line=dict(width=lw, color=col),
                hovertext=etxt, hoverinfo="text",
                hoverlabel=dict(bgcolor="white", font_size=11),
                showlegend=False,
            ))

    top_edges = sorted(G.edges(data=True), key=lambda e: e[2].get("weight", 0), reverse=True)
    lx, ly, ltxt = [], [], []
    for u, v, data in top_edges[:_EDGE_LABEL_TOP_N]:
        x0, y0 = pos[u]; x1, y1 = pos[v]
        lx.append((x0 + x1) / 2)
        ly.append((y0 + y1) / 2)
        ltxt.append(str(data.get("weight", "")))
    if lx:
        traces.append(go.Scatter(
            x=lx, y=ly, mode="text", text=ltxt,
            textfont=dict(size=8, color="#555555"),
            hoverinfo="none", showlegend=False,
        ))

    community_buckets: dict[int, dict] = {}
    enrich = enrichment or {}

    for nid, attrs in G.nodes(data=True):
        cid   = attrs.get("community", 0)
        deg   = degrees.get(nid, 0)
        npt   = attrs.get("npt_ratio", 0.0)
        label = attrs.get("label", nid)
        ntype = attrs.get("node_type", "term")
        size  = 11 + 30 * (deg / max(n_max_deg, 1))

        if npt > 0.70:
            border_col, border_w = "#D32F2F", 3.5
        elif npt > 0.40:
            border_col, border_w = "#F57C00", 2.5
        else:
            border_col, border_w = "#FFFFFF", 1.0

        e = enrich.get(nid, {})

        npt_flag  = ("⚠ High NPT association" if npt > 0.70
                     else ("△ Elevated NPT exposure" if npt > 0.40 else ""))
        risk_line = (f"{npt_flag}<br>" if npt_flag else "") + f"NPT {npt:.0%} · {deg} connections"

        n_ops     = e.get("n_ops",     attrs.get("count", 0))
        total_hrs = e.get("total_hrs", "?")
        npt_hrs   = e.get("npt_hrs",   "?")
        scale_line = f"{n_ops} ops · {total_hrs}h total · {npt_hrs}h NPT"

        temp_parts = []
        n_days  = e.get("n_days", "")
        temporal = e.get("temporal_note", "")
        if n_days:
            temp_parts.append(f"Seen on {n_days} reporting days")
        if temporal:
            if " · peak: " in temporal:
                active_part, peak_part = temporal.split(" · peak: ", 1)
                temp_parts += [active_part, f"Peak activity: {peak_part}"]
            else:
                temp_parts.append(temporal)
        temporal_html = "<br>".join(temp_parts)

        neighbours = sorted(
            G[nid].items(), key=lambda x: x[1].get("weight", 0), reverse=True
        )[:5]
        linked_html = (
            "<br>".join(f"  {nb} ({d.get('weight', 0)})" for nb, d in neighbours)
            if neighbours else "—"
        )

        excerpts     = e.get("excerpts", [])
        excerpt_html = "<br>".join(f"• {ex}" for ex in excerpts) if excerpts else ""

        parts = [f"<b>{label}</b>", risk_line, "─" * 24, scale_line]
        if temporal_html:
            parts += ["─" * 24, temporal_html]
        if linked_html:
            parts += ["─" * 24, linked_html]
        if excerpt_html:
            parts += ["─" * 24, excerpt_html]

        if cid not in community_buckets:
            community_buckets[cid] = dict(x=[], y=[], hover=[], size=[],
                                          symbol=[], bc=[], bw=[], text=[])
        cb = community_buckets[cid]
        cb["x"].append(pos[nid][0]); cb["y"].append(pos[nid][1])
        cb["hover"].append("<br>".join(parts))
        cb["size"].append(size)
        cb["symbol"].append("diamond" if ntype == "op_code" else "circle")
        cb["bc"].append(border_col); cb["bw"].append(border_w)
        cb["text"].append(label if nid in label_set else "")

    for cid, cb in sorted(community_buckets.items()):
        traces.append(go.Scatter(
            x=cb["x"], y=cb["y"],
            mode="markers+text",
            marker=dict(
                size=cb["size"],
                color=_COMMUNITY_COLOURS.get(cid, "#999"),
                symbol=cb["symbol"],
                line=dict(color=cb["bc"], width=cb["bw"]),
                opacity=0.92,
            ),
            text=cb["text"], textposition="top center",
            textfont=dict(size=9, color="#1A1A1A"),
            hovertext=cb["hover"], hoverinfo="text",
            name=f"Workflow {cid + 1}", showlegend=True,
        ))

    return go.Figure(data=traces, layout=go.Layout(
        paper_bgcolor="#FAFAFA", plot_bgcolor="#FAFAFA",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(orientation="v", x=1.01, y=1,
                    font=dict(size=11), title=dict(text="Workflow")),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest", height=540,
        annotations=[dict(
            text=(
                "◆ = operation category (structured DDR code)   "
                "● = operational term (from description text)   "
                "border: 🔴 high NPT · 🟠 medium · ⚪ low"
            ),
            showarrow=False, xref="paper", yref="paper",
            x=0.01, y=-0.02, xanchor="left",
            font=dict(size=9, color="#888"),
        )],
    ))


def _infer_community_name(top_nodes: list[str]) -> str:
    kw = " ".join(top_nodes[:6]).lower()
    if any(t in kw for t in ["fishing", "magnet", "junk", "mill", "stacey", "htpac"]):
        return "Fishing & Recovery"
    if any(t in kw for t in ["cable", "profibus", "prs", "drag chain", "rider"]):
        return "Rig Equipment"
    if any(t in kw for t in ["frac", "sleeve", "ncs", "packer", "flow-in-area", "stim"]):
        return "Stimulation & Completion"
    if any(t in kw for t in ["casing", "liner", "cement", "centralizer"]):
        return "Casing & Cementing"
    if any(t in kw for t in ["torque", "rop", "pick-up-weight", "slack-off", "parameters"]):
        return "Drilling Parameters"
    if any(t in kw for t in ["rih", "pooh", "drill-pipe", "bha", "top-drive", "make-up"]):
        return "Tripping & Assembly"
    if any(t in kw for t in ["weather", "vessel", "crane", "aukra", "skandi", "tide"]):
        return "Logistics & Marine"
    if any(t in kw for t in ["hydraulic", "position", "tilt", "valaris", "rig system"]):
        return "Rig Systems"
    if any(t in kw for t in ["sbp", "choke", "mpd", "flowrate", "surface-back", "ecd"]):
        return "Pressure Management"
    if any(t in kw for t in ["pressure", "pump", "flow", "test", "annulus"]):
        return "Pressure & Circulation"
    return "Mixed Operations"


def _community_cards(graph_data: dict) -> None:
    nodes_df = pd.DataFrame(graph_data["nodes"])
    if nodes_df.empty:
        return

    for cid in sorted(nodes_df["community"].unique()):
        members    = nodes_df[nodes_df["community"] == cid].sort_values("degree", ascending=False)
        top_labels = members["label"].head(8).tolist()
        avg_npt    = float(
            (members["npt_ratio"] * members["degree"]).sum()
            / max(members["degree"].sum(), 1)
        )
        top5_counts    = members["count"].head(5)
        median_support = int(top5_counts.median()) if len(top5_counts) else 0
        low_support    = median_support < 15
        risk_node      = members.sort_values("npt_ratio", ascending=False).iloc[0]
        n_name         = _infer_community_name([n for n in top_labels if not n.startswith("op:")])

        if avg_npt > 0.65:
            npt_badge = "🔴 High NPT association"
        elif avg_npt > 0.35:
            npt_badge = "🟠 Mixed NPT"
        else:
            npt_badge = "🟢 Mostly productive"

        with st.expander(f"**{n_name}** — {npt_badge}", expanded=True):
            c1, c2, c3 = st.columns([4, 1, 1])
            with c1:
                terms = [
                    f"**{lbl}**" if lbl == risk_node["label"] else lbl
                    for lbl in top_labels
                ]
                st.markdown("**Key terms:** " + " · ".join(terms))
            with c2:
                st.metric("NPT rate", f"{avg_npt:.0%}",
                          help="Weighted average NPT ratio across terms in this workflow.")
            with c3:
                st.metric("Evidence", f"~{median_support} ops",
                          help="Median term frequency across the top 5 nodes.")
            if low_support:
                st.caption(
                    "⚠ Small sample — fewer than 15 operations reference these terms. "
                    "Interpret this workflow with caution until more data is available."
                )
            elif avg_npt > 0.35:
                st.caption(
                    f"Most frequently observed during NPT: **{risk_node['label']}** "
                    f"— {risk_node['npt_ratio']:.0%} NPT rate "
                    f"across {int(risk_node['count'])} occurrences "
                    f"({int(risk_node['degree'])} connections)"
                )


def _render_kpis(graph_data: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Operations",           graph_data["n_ops"])
    c2.metric("Days",                 graph_data["n_docs"])
    c3.metric("NPT",                  f"{graph_data['npt_pct']}%")
    c4.metric("Operational workflows", graph_data["n_communities"])


def _render_tab_summary(graph_data: dict) -> None:
    st.markdown(
        "Each workflow below represents a distinct pattern of operations identified "
        "from language patterns across all reports for this phase."
    )
    _community_cards(graph_data)

    st.divider()
    st.subheader("Strongest operational relationships")
    st.caption(
        "Pairs that co-occur most frequently — these are the core "
        "workflow connections for this phase."
    )
    edges_df = pd.DataFrame(graph_data["links"])
    if not edges_df.empty:
        for col in ["source", "target"]:
            edges_df[col] = edges_df[col].apply(
                lambda x: label_op_code(x[3:]) if str(x).startswith("op:") else x
            )
        top_edges = (
            edges_df.rename(columns={"source": "Term A", "target": "Term B",
                                      "weight": "Co-occurrences"})
            [["Term A", "Term B", "Co-occurrences"]]
            .head(15)
        )
        st.dataframe(top_edges, hide_index=True, use_container_width=True)


def _render_tab_focus(graph_data: dict, enrichment: dict) -> None:
    st.markdown(
        "Select what you want to understand. The graph shows only the "
        "relevant terms and their strongest connections."
    )

    mode_labels = {k: v[0] for k, v in _GRAPH_MODES.items()}
    mode_key = st.radio(
        "What do you want to understand?",
        list(mode_labels.keys()),
        format_func=lambda k: mode_labels[k],
        horizontal=True,
    )

    all_weights = [e.get("norm_weight", 0) for e in graph_data["links"]]
    default_threshold = float(pd.Series(all_weights).quantile(0.85)) if all_weights else 0.0

    min_nw = st.slider(
        "Edge strength threshold",
        min_value=0.0, max_value=1.0,
        value=round(default_threshold, 2), step=0.05,
        help="Only show edges stronger than this value. Recommended: 0.7–0.9.",
    )
    max_labels = st.slider("Labels to show", 3, 20, 8, 1,
                           help="Only the top N most-connected nodes are labelled.")

    with st.spinner("Rendering..."):
        fig = render_graph(graph_data, min_norm_weight=min_nw,
                           mode_key=mode_key, max_labels=max_labels,
                           enrichment=enrichment)
    st.plotly_chart(_apply_chart_theme(fig), use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "**◆ Diamond** = operation category (structured DDR code). "
        "**● Circle** = operational term from description text. "
        "Node size = number of connections. Border: 🔴 high NPT (>70%) · 🟠 medium · ⚪ low."
    )
    if mode_key != "all":
        st.info(
            f"**{mode_labels[mode_key]}** — showing seed nodes matching this "
            "mode plus their immediate neighbours. Nodes outside this focus are hidden."
        )


def _render_tab_engineering(graph_data: dict, enrichment: dict) -> None:
    st.markdown(
        "Full graph for engineering investigation. "
        "Adjust the threshold to explore connection density."
    )

    eng_threshold = st.slider(
        "Minimum edge strength (engineering)",
        min_value=0.0, max_value=1.0,
        value=0.50, step=0.05,
        help="Recommended: 0.5. Lower = more connections; higher = strongest pairs only.",
        key="eng_threshold",
    )
    eng_labels = st.slider("Labels", 5, 30, 12, 1, key="eng_labels")

    with st.spinner("Rendering full graph..."):
        fig_eng = render_graph(graph_data, min_norm_weight=eng_threshold,
                               mode_key="all", max_labels=eng_labels,
                               enrichment=enrichment)
    st.plotly_chart(_apply_chart_theme(fig_eng), use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "**◆ Diamond** = operation category (structured DDR code). "
        "**● Circle** = operational term from description text. "
        "Border: 🔴 >70% NPT · 🟠 40–70% · ⚪ low. "
        "Node size = connections. Edge thickness = relationship strength."
    )

    st.subheader("All nodes — ranked by connectivity")
    nodes_df = pd.DataFrame(graph_data["nodes"])
    if nodes_df.empty:
        st.info("No nodes for this phase.")
        return
    nodes_df["npt_ratio"] = nodes_df["npt_ratio"].map("{:.0%}".format)
    nodes_df["betweenness"] = nodes_df.get("betweenness", 0).map(
        lambda x: f"{x:.3f}" if isinstance(x, float) else "—"
    )
    nodes_disp = (
        nodes_df[["label", "degree", "count", "npt_ratio", "community"]]
        .rename(columns={"label": "Term", "degree": "Connections",
                         "count": "Frequency", "npt_ratio": "NPT ratio",
                         "community": "Workflow"})
        .sort_values("Connections", ascending=False)
    )
    st.dataframe(nodes_disp, hide_index=True, use_container_width=True, height=400)


def _render_tab_by_op_code(ops: pd.DataFrame) -> None:
    st.markdown(
        "Hours and NPT by structured operation type, taken directly from the DDR "
        "Operation Summary tables (`op_code`) — across the **full well**, not filtered "
        "to the phase selected above."
    )

    df = ops.copy()
    code = df["op_code"].astype(str).str.strip()
    df["op_code_display"] = code.where(code != "", "(unclassified)").map(
        lambda c: label_op_code(c) if c != "(unclassified)" else c
    )

    total_by_code = df.groupby("op_code_display")["duration_hr"].sum()
    npt_by_code   = (
        df.loc[df["is_npt"]].groupby("op_code_display")["duration_hr"].sum()
    )
    count_by_code = df.groupby("op_code_display").size()

    summary = pd.DataFrame({"total_h": total_by_code, "n_ops": count_by_code})
    summary["npt_h"] = npt_by_code.reindex(summary.index).fillna(0.0)
    summary["prod_h"] = summary["total_h"] - summary["npt_h"]
    summary["npt_pct"] = (summary["npt_h"] / summary["total_h"].replace(0, 1) * 100)
    summary = summary.sort_values("total_h", ascending=False)

    if "(unclassified)" in summary.index:
        unclassified_pct = 100 * summary.loc["(unclassified)", "total_h"] / summary["total_h"].sum()
        if unclassified_pct >= 5:
            st.caption(
                f"⚠ {unclassified_pct:.0f}% of well time has no structured operation type "
                "in the source DDRs (shown as \"(unclassified)\")."
            )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=summary.index, x=summary["prod_h"], name="Productive",
        orientation="h", marker_color="#4CAF50", opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        y=summary.index, x=summary["npt_h"], name="NPT",
        orientation="h", marker_color="#F44336", opacity=0.85,
    ))
    fig.update_layout(
        barmode="stack", height=max(320, len(summary) * 32 + 80),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Hours", title="Hours by Operation Type",
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
    )
    st.plotly_chart(_apply_chart_theme(fig), use_container_width=True, config={"displayModeBar": False})

    disp = summary.reset_index().rename(columns={
        "op_code_display": "Operation Type",
        "total_h": "Total h", "npt_h": "NPT h",
        "npt_pct": "NPT %", "n_ops": "Count",
    })[["Operation Type", "Total h", "NPT h", "NPT %", "Count"]]
    disp["Total h"] = disp["Total h"].map("{:.0f}h".format)
    disp["NPT h"]   = disp["NPT h"].map("{:.0f}h".format)
    disp["NPT %"]   = disp["NPT %"].map("{:.0f}%".format)
    st.caption("Ranked by total hours")
    st.dataframe(disp, hide_index=True, use_container_width=True)


def page_operational_graph(ops: pd.DataFrame) -> None:
    st.header("Operational Analysis")

    st.subheader("By Operation Type — full well")
    _render_tab_by_op_code(ops)
    st.divider()

    st.subheader("Text-mined workflow patterns — by phase")
    st.caption(
        "The sections below infer workflow patterns from operation *text* rather than "
        "the structured op_code field above — useful for the ~10% of rows with no "
        "structured operation type, but treat as exploratory, not authoritative."
    )

    phases = sorted(
        ops["phase"].dropna().unique(),
        key=lambda x: PHASE_ORDER.index(x) if x in PHASE_ORDER else 99,
    )
    c1, c2 = st.columns([3, 2])
    with c1:
        _default_phase = next(
            (p for p in ("Production Drilling", "PROD1") if p in phases), None
        )
        phase = st.selectbox(
            "Drilling phase", phases, format_func=label_phase,
            index=phases.index(_default_phase) if _default_phase else 0,
            key="op_graph_phase",
        )
    with c2:
        window = st.radio(
            "Relationship type",
            ["Same row", "Sequential (±2 rows)"], horizontal=True,
            help="Same row: strict co-occurrence. Sequential: captures what leads to what.",
        )

    graph_data = load_graph(phase, window)
    if not graph_data:
        st.warning("Graph not found. Run scripts/build_graphs.py first.")
        return

    with st.spinner("Loading operational context..."):
        enrichment = build_node_enrichment(phase, window)

    _render_kpis(graph_data)
    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📋 Operational Summary",
        "🔍 Focus Graph",
        "🔬 Engineering View",
    ])
    with tab1:
        _render_tab_summary(graph_data)
    with tab2:
        _render_tab_focus(graph_data, enrichment)
    with tab3:
        _render_tab_engineering(graph_data, enrichment)
