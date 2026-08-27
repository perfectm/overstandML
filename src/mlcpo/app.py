"""ML CPO web interface — Plotly Dash, mirroring the reference system's stack.

Two entry modes:
  python -m mlcpo.app --refresh data/<ts_export>.csv   # recompute caches (slow)
  python -m mlcpo.app                                  # serve from caches

Serving: 127.0.0.1:8050 by default; --host 0.0.0.0 (or MLCPO_HOST/MLCPO_PORT)
to expose on the LAN / behind a reverse proxy (e.g. closet.cottonmike.com).

Charts follow the repo's dataviz conventions: light pink surface #fff0f5, one
axis per chart, 2px lines, hairline grid, fixed categorical order
(children S1..S6 = slots 1..6; streams: ML=blue, Baseline=orange,
Best Child=aqua, Ensemble=yellow), legends + direct end labels.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

CACHE = Path("data/app_cache")

# validated reference palette (light mode, pink)
SURFACE, PAGE = "#fff0f5", "#ffe0eb"
INK, INK2, MUTED, GRID, BASELINE_INK = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STREAM_COLORS = {
    "ML (P3_v2_2)": SLOTS[0],
    "BASELINE": SLOTS[1],
    "BEST CHILD": SLOTS[2],
    "ENSEMBLE (vote)": SLOTS[3],
}
CHILD_COLORS = {f"S{i+1}": SLOTS[i] for i in range(6)}


# ------------------------------------------------------------------ caching

def refresh_cache(ts_csv: str, sets: list[str] | None = None) -> None:
    """Recompute everything the app serves. Slow (runs full walk-forwards)."""
    from .data.children import build_style_children
    from .data.ts_to_oo import convert_file
    from .diagnostics import metrics, parent_lab
    from .live import DEFAULT_PARTITION_PATH, apply_partition, load_partition, save_partition
    from .model.ensemble import run_ensemble
    from .model.hp import load_hp_sets
    from .model.walkforward import WalkForwardConfig, build_dataset
    from .features.market import build_market_features, fetch_ohlc

    sets = sets or ["P3_v2_2", "optuna_s6_pnl", "optuna_s6_mar"]
    CACHE.mkdir(parents=True, exist_ok=True)
    oo = convert_file(ts_csv)

    if DEFAULT_PARTITION_PATH.exists():
        parent = apply_partition(oo, load_partition())
    else:
        parent, assignment = build_style_children(oo, n_children=6)
        save_partition(assignment)
        parent = apply_partition(oo, load_partition())

    pnl = parent.daily_pnl()
    feats = build_market_features(fetch_ohlc(str(pnl.index.min().date()), None))
    ds = build_dataset(pnl, feats)

    hp = load_hp_sets()
    cfg = WalkForwardConfig(top_k=1)
    ens, per_set = run_ensemble(ds, {n: hp[n] for n in sets}, cfg, method="vote")
    primary = per_set[sets[0]]

    streams = pd.DataFrame(
        {
            "ML (P3_v2_2)": primary.stream("ml"),
            "BASELINE": primary.stream("baseline"),
            "BEST CHILD": primary.stream("best_child"),
            "ENSEMBLE (vote)": ens.stream("ml"),
        }
    )
    streams.to_parquet(CACHE / "streams.parquet")

    cycles = ens.cycles.copy()
    cycles["pick"] = cycles["picks"].map(lambda p: p[0])
    cycles[["pick", "ml_pnl", "baseline_pnl", "best_child_pnl"]].to_parquet(CACHE / "cycles.parquet")

    summary = {name: metrics.summary(streams[name]) for name in streams.columns}
    (CACHE / "summary.json").write_text(json.dumps(summary, indent=2))

    report = parent_lab.parent_report(parent)
    children = {
        c.name: {
            "strategies": sorted(c.strategies),
            **report["per_child"].loc[c.name].round(4).to_dict(),
        }
        for c in parent.children
    }
    (CACHE / "children.json").write_text(json.dumps(children, indent=2))

    (CACHE / "meta.json").write_text(
        json.dumps(
            {
                "refreshed_at": str(pd.Timestamp.now().round("s")),
                "ts_csv": str(ts_csv),
                "data_range": [str(pnl.index.min().date()), str(pnl.index.max().date())],
                "n_cycles": int(ens.n_cycles),
                "hp_sets": sets,
                "diagnostics": {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in report.items()
                    if k != "per_child"
                },
            },
            indent=2,
        )
    )


def load_cache() -> dict:
    if not (CACHE / "meta.json").exists():
        raise SystemExit("no app cache — run:  python -m mlcpo.app --refresh data/<ts_export>.csv")
    latest_decision = None
    live_dir = Path("data/live")
    if live_dir.exists():
        files = sorted(live_dir.glob("*.json"))
        if files:
            latest_decision = json.loads(files[-1].read_text())
    return {
        "meta": json.loads((CACHE / "meta.json").read_text()),
        "summary": json.loads((CACHE / "summary.json").read_text()),
        "children": json.loads((CACHE / "children.json").read_text()),
        "streams": pd.read_parquet(CACHE / "streams.parquet"),
        "cycles": pd.read_parquet(CACHE / "cycles.parquet"),
        "decision": latest_decision,
    }


# ------------------------------------------------------------------ figures

def _frame(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK2, size=12),
        margin=dict(l=56, r=110, t=48, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom", font=dict(color=INK2)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE_INK, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE_INK, tickfont=dict(color=MUTED),
                     zeroline=True, zerolinecolor=BASELINE_INK, tickformat="$,.0f")
    return fig


def equity_fig(streams: pd.DataFrame):
    import plotly.graph_objects as go

    fig = go.Figure()
    for name in STREAM_COLORS:
        eq = streams[name].cumsum()
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=name, mode="lines",
                                 line=dict(color=STREAM_COLORS[name], width=2)))
        fig.add_annotation(x=eq.index[-1], y=eq.iloc[-1], text=f" {name.split(' ')[0]}",
                           showarrow=False, xanchor="left", font=dict(color=STREAM_COLORS[name], size=11))
    return _frame(fig, "Cumulative P&L — walk-forward streams")


def drawdown_fig(streams: pd.DataFrame):
    import plotly.graph_objects as go

    fig = go.Figure()
    for name in ("ML (P3_v2_2)", "ENSEMBLE (vote)"):
        eq = streams[name].cumsum()
        dd = eq - eq.cummax()
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=name, mode="lines",
                                 line=dict(color=STREAM_COLORS[name], width=2)))
    return _frame(fig, "Drawdown from running peak")


def picks_fig(cycles: pd.DataFrame):
    import plotly.graph_objects as go

    fig = go.Figure()
    for child in sorted(cycles["pick"].unique()):
        sel = cycles[cycles["pick"] == child]
        fig.add_trace(go.Scatter(x=sel.index, y=sel["pick"], name=child, mode="markers",
                                 marker=dict(color=CHILD_COLORS.get(child, MUTED), size=8)))
    fig = _frame(fig, "Daily pick timeline")
    fig.update_yaxes(tickformat=None, categoryorder="category descending")
    return fig


def child_pnl_fig(cycles: pd.DataFrame):
    import plotly.graph_objects as go

    contrib = cycles.groupby("pick")["ml_pnl"].sum().sort_index()
    fig = go.Figure(
        go.Bar(x=contrib.index, y=contrib.values,
               marker=dict(color=[CHILD_COLORS.get(c, MUTED) for c in contrib.index],
                           cornerradius=4),
               text=[f"${v:,.0f}" for v in contrib.values], textposition="outside",
               textfont=dict(color=INK2, size=11))
    )
    fig = _frame(fig, "Ensemble P&L contribution by picked child")
    fig.update_layout(showlegend=False, margin=dict(r=24))
    return fig


# ------------------------------------------------------------------- layout

def tile(label, value, sub=""):
    from dash import html

    return html.Div(
        [html.Div(label, style={"color": MUTED, "fontSize": "12px"}),
         html.Div(value, style={"color": INK, "fontSize": "26px", "fontWeight": 600}),
         html.Div(sub, style={"color": INK2, "fontSize": "12px"})],
        style={"background": SURFACE, "border": f"1px solid {GRID}", "borderRadius": "8px",
               "padding": "14px 18px", "flex": "1"},
    )


def build_app(cache: dict):
    from dash import Dash, dcc, html

    meta, summary = cache["meta"], cache["summary"]
    ml, ens = summary["ML (P3_v2_2)"], summary["ENSEMBLE (vote)"]
    base = summary["BASELINE"]

    tiles = html.Div(
        [
            tile("ML total P&L", f"${ml['total_pnl']:,.0f}", f"baseline ${base['total_pnl']:,.0f}"),
            tile("Ensemble total P&L", f"${ens['total_pnl']:,.0f}", "vote of 3 HP sets"),
            tile("ML Sharpe", f"{ml['sharpe']:.2f}", f"baseline {base['sharpe']:.2f}"),
            tile("ML MaxDD", f"${ml['max_dd']:,.0f}", f"{ml['max_dd_pct']:.1%} · {ml['max_dd_days']:.0f}d"),
            tile("ML MAR", f"{ml['mar']:.2f}", f"CAGR {ml['cagr']:.1%}"),
            tile("Cycles", f"{meta['n_cycles']}", f"{meta['data_range'][0]} → {meta['data_range'][1]}"),
        ],
        style={"display": "flex", "gap": "12px", "margin": "16px 0"},
    )

    children_blocks = []
    for name, info in cache["children"].items():
        children_blocks.append(
            html.Div(
                [
                    html.Div([html.Span("● ", style={"color": CHILD_COLORS.get(name, MUTED)}),
                              html.B(name, style={"color": INK}),
                              html.Span(f"  ${info['realized_pnl']:,.0f} · Sharpe {info['sharpe']:.2f}"
                                        f" · picked {info['winner_pct']:.0%} of days",
                                        style={"color": INK2, "fontSize": "13px"})]),
                    html.Ul([html.Li(s, style={"color": INK2, "fontSize": "12px"})
                             for s in info["strategies"]]),
                ],
                style={"background": SURFACE, "border": f"1px solid {GRID}",
                       "borderRadius": "8px", "padding": "12px 16px", "marginBottom": "10px"},
            )
        )

    if cache["decision"]:
        d = cache["decision"]
        live_tab = html.Div(
            [
                html.H3(f"Decision for {d['date']}", style={"color": INK}),
                html.Div(f"PICK: {', '.join(d['picks'])}",
                         style={"fontSize": "28px", "fontWeight": 700,
                                "color": CHILD_COLORS.get(d["picks"][0], INK)}),
                html.P(f"method {d['method']} · top-k {d['top_k']}", style={"color": INK2}),
                html.Table(
                    [html.Tr([html.Th("child"), html.Th("combined score")])] +
                    [html.Tr([html.Td(c), html.Td(f"{v:,.0f}")])
                     for c, v in sorted(d["combined_scores"].items(), key=lambda kv: -kv[1])],
                    style={"color": INK2},
                ),
                html.H4("TS Bot Portfolios — verify by hand before market", style={"color": INK}),
                html.Ul(
                    [html.Li(f"ENABLE {c} ({len(s)} bots)", style={"color": INK, "fontWeight": 600})
                     for c, s in d["enable"].items()] +
                    [html.Li(f"disable {c} ({len(s)} bots)", style={"color": MUTED})
                     for c, s in d["disable"].items()]
                ),
            ]
        )
    else:
        live_tab = html.P("No decision file yet — run:  python -m mlcpo.live data/<ts_export>.csv",
                          style={"color": INK2})

    app = Dash(__name__, title="ML CPO")
    tab_style = {"background": PAGE, "border": "none", "color": INK2, "padding": "10px 16px"}
    sel_style = {**tab_style, "color": INK, "fontWeight": 600, "borderBottom": f"2px solid {SLOTS[0]}",
                 "background": SURFACE}
    app.layout = html.Div(
        [
            html.Div(
                [html.H2("ML CPO — daily child-portfolio selection", style={"color": INK, "margin": 0}),
                 html.Div(f"data {meta['data_range'][0]} → {meta['data_range'][1]} · "
                          f"sets {', '.join(meta['hp_sets'])} · refreshed {meta['refreshed_at']}",
                          style={"color": MUTED, "fontSize": "12px"})],
                style={"padding": "18px 0 6px"},
            ),
            tiles,
            dcc.Tabs(
                [
                    dcc.Tab(label="Overview", style=tab_style, selected_style=sel_style,
                            children=[dcc.Graph(figure=equity_fig(cache["streams"])),
                                      dcc.Graph(figure=drawdown_fig(cache["streams"]))]),
                    dcc.Tab(label="Cycles", style=tab_style, selected_style=sel_style,
                            children=[dcc.Graph(figure=picks_fig(cache["cycles"])),
                                      dcc.Graph(figure=child_pnl_fig(cache["cycles"]))]),
                    dcc.Tab(label="Children", style=tab_style, selected_style=sel_style,
                            children=[html.Div(children_blocks, style={"padding": "16px 0"})]),
                    dcc.Tab(label="Live", style=tab_style, selected_style=sel_style,
                            children=[html.Div(live_tab, style={"padding": "16px 0"})]),
                ],
                style={"marginTop": "4px"},
            ),
        ],
        style={"maxWidth": "1080px", "margin": "0 auto", "padding": "0 20px 40px",
               "background": PAGE, "minHeight": "100vh",
               "fontFamily": 'system-ui, -apple-system, "Segoe UI", sans-serif'},
    )
    return app


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="ML CPO web interface")
    ap.add_argument("--refresh", metavar="TS_CSV", help="recompute caches from a TS export, then exit")
    ap.add_argument("--host", default=os.environ.get("MLCPO_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MLCPO_PORT", "8050")))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    if args.refresh:
        refresh_cache(args.refresh)
        print(f"cache refreshed under {CACHE}/")
        return

    build_app(load_cache()).run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
