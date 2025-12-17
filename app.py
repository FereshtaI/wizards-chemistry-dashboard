from __future__ import annotations

from dash import Dash, dcc, html, Input, Output, State
from dash import dash_table
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

DATA_PATH = "data/wizards_lineups_2025_26.csv"

app = Dash(__name__)
app.title = "Wizards Chemistry Dashboard"

# ----------------------------
# Data helpers
# ----------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().upper() for c in df.columns]

    required = {"GROUP_NAME", "MIN", "NET_RATING"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. You have: {list(df.columns)}")

    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0.0)
    df["NET_RATING"] = pd.to_numeric(df["NET_RATING"], errors="coerce").fillna(0.0)

    # chemistry proxy
    df["CHEMISTRY_SCORE"] = df["NET_RATING"]

    # per minute (avoid divide by 0)
    df["CHEM_PER_MIN"] = np.where(df["MIN"] > 0, df["CHEMISTRY_SCORE"] / df["MIN"], 0.0)

    df["CHEM_TIER"] = pd.cut(
        df["CHEMISTRY_SCORE"],
        bins=[-1e9, -5, 5, 1e9],
        labels=["Low", "Medium", "High"],
    )

    # pseudo time order (if you don't have games/dates):
    # we sort by MIN ascending so "later samples" have more usage,
    # and create a sample index. Label this clearly as "Rolling Sample".
    df = df.sort_values("MIN", ascending=True).reset_index(drop=True)
    df["SAMPLE_IDX"] = np.arange(len(df))

    return df


df = load_data()


def extract_players(group_name: str) -> list[str]:
    if not isinstance(group_name, str):
        return []
    return [p.strip() for p in group_name.split(" - ") if p.strip()]


ALL_PLAYERS = sorted({p for g in df["GROUP_NAME"].dropna().tolist() for p in extract_players(g)})


def lineup_contains_all(group_name: str, selected_players: list[str]) -> bool:
    if not selected_players:
        return True
    return set(selected_players).issubset(set(extract_players(group_name)))


def short_name(s: str, max_len: int = 60) -> str:
    if not isinstance(s, str):
        return ""
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def build_table(rows: list[dict], columns: list[str], page_size: int = 8, tooltip_cols: set[str] | None = None):
    tooltip_cols = tooltip_cols or set()
    tooltip_data = []
    for r in rows:
        trow = {}
        for c in tooltip_cols:
            if c in r and r[c] is not None:
                trow[c] = {"value": str(r[c]), "type": "text"}
        tooltip_data.append(trow)

    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in columns],
        page_size=page_size,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#0f172a",
            "color": "#eaeef6",
            "fontWeight": "700",
            "border": "1px solid #1f2a44",
        },
        style_cell={
            "backgroundColor": "#121a2b",
            "color": "#eaeef6",
            "border": "1px solid #1f2a44",
            "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Arial",
            "fontSize": "12px",
            "padding": "8px",
            "whiteSpace": "normal",
            "height": "auto",
            "maxWidth": "560px",
        },
        tooltip_data=tooltip_data if tooltip_cols else None,
        tooltip_duration=None,
    )


# ----------------------------
# Analytics helpers
# ----------------------------
def compute_actions(dff: pd.DataFrame):
    if dff.empty:
        return [], [], []

    chem_hi = float(dff["CHEMISTRY_SCORE"].quantile(0.75))
    chem_lo = float(dff["CHEMISTRY_SCORE"].quantile(0.25))
    min_hi = float(dff["MIN"].quantile(0.75))
    min_lo = float(dff["MIN"].quantile(0.35))
    permin_hi = float(dff["CHEM_PER_MIN"].quantile(0.75))

    expand_df = dff[(dff["CHEMISTRY_SCORE"] >= chem_hi) & (dff["MIN"] <= min_lo)].copy()
    reduce_df = dff[(dff["CHEMISTRY_SCORE"] <= chem_lo) & (dff["MIN"] >= min_hi)].copy()
    test_df = dff[(dff["CHEM_PER_MIN"] >= permin_hi) & (dff["MIN"] <= min_lo)].copy()

    def pack(df_in, topn=8):
        if df_in.empty:
            return []
        out = (
            df_in.sort_values(["CHEMISTRY_SCORE", "MIN"], ascending=[False, False])
            .head(topn)[["GROUP_NAME", "MIN", "CHEMISTRY_SCORE", "CHEM_PER_MIN", "CHEM_TIER"]]
            .copy()
        )
        out["LINEUP"] = out["GROUP_NAME"].apply(short_name)
        out["MIN"] = out["MIN"].round(1)
        out["CHEMISTRY_SCORE"] = out["CHEMISTRY_SCORE"].round(2)
        out["CHEM_PER_MIN"] = out["CHEM_PER_MIN"].round(3)
        out = out.drop(columns=["GROUP_NAME"])
        return out.to_dict("records")

    return pack(expand_df), pack(test_df), pack(reduce_df)


def compute_top_teammates(dff: pd.DataFrame, selected_players: list[str], topn=12):
    if not selected_players or dff.empty:
        return []
    sel = set(selected_players)

    rows = []
    for _, r in dff.iterrows():
        players = set(extract_players(r["GROUP_NAME"]))
        if not sel.issubset(players):
            continue
        for p in players - sel:
            rows.append({"PLAYER": p, "MIN": float(r["MIN"]), "CHEMISTRY_SCORE": float(r["CHEMISTRY_SCORE"])})

    if not rows:
        return []

    tmp = pd.DataFrame(rows)
    agg = (
        tmp.groupby("PLAYER", as_index=False)
        .agg(
            LINEUP_MIN=("MIN", "sum"),
            AVG_CHEM=("CHEMISTRY_SCORE", "mean"),
            COUNT_LINEUPS=("PLAYER", "count"),
        )
    )
    agg["LINEUP_MIN"] = agg["LINEUP_MIN"].round(1)
    agg["AVG_CHEM"] = agg["AVG_CHEM"].round(2)
    agg = agg.sort_values(["AVG_CHEM", "LINEUP_MIN"], ascending=[False, False]).head(topn)
    return agg.to_dict("records")


def compute_best_builds(df_in: pd.DataFrame, focus_player: str, min_threshold: float, topn: int = 3):
    if not focus_player or df_in.empty:
        return []
    dff = df_in[df_in["MIN"] >= (min_threshold or 0)].copy()
    dff = dff[dff["GROUP_NAME"].apply(lambda g: focus_player in extract_players(g))].copy()
    if dff.empty:
        return []
    dff["WEIGHTED_SCORE"] = dff["CHEMISTRY_SCORE"] * dff["MIN"]
    top = dff.sort_values(["WEIGHTED_SCORE", "MIN"], ascending=[False, False]).head(topn).copy()
    top["LINEUP"] = top["GROUP_NAME"].apply(short_name)
    out = top[["LINEUP", "MIN", "CHEMISTRY_SCORE", "CHEM_PER_MIN", "WEIGHTED_SCORE", "CHEM_TIER"]].copy()
    out["MIN"] = out["MIN"].round(1)
    out["CHEMISTRY_SCORE"] = out["CHEMISTRY_SCORE"].round(2)
    out["CHEM_PER_MIN"] = out["CHEM_PER_MIN"].round(3)
    out["WEIGHTED_SCORE"] = out["WEIGHTED_SCORE"].round(1)
    return out.to_dict("records")


def stability_index(dff: pd.DataFrame):
    """Avg NET vs variance NET per lineup."""
    if dff.empty:
        return pd.DataFrame(columns=["GROUP_NAME", "AVG_NET", "VAR_NET", "MIN_SUM"])
    agg = (
        dff.groupby("GROUP_NAME", as_index=False)
        .agg(
            AVG_NET=("CHEMISTRY_SCORE", "mean"),
            VAR_NET=("CHEMISTRY_SCORE", "var"),
            MIN_SUM=("MIN", "sum"),
        )
    )
    agg["VAR_NET"] = agg["VAR_NET"].fillna(0.0)
    return agg


def on_off_lift(df_in: pd.DataFrame, min_threshold: float):
    """For each player: avg net with player vs without player. (Approx using lineup rows.)"""
    base = df_in[df_in["MIN"] >= (min_threshold or 0)].copy()
    if base.empty:
        return pd.DataFrame(columns=["PLAYER", "WITH_NET", "WITHOUT_NET", "DELTA"])

    # precompute lineup players sets
    lineup_players = base["GROUP_NAME"].apply(lambda g: set(extract_players(g)))
    base = base.assign(_PLAYERS=lineup_players)

    overall = float(base["CHEMISTRY_SCORE"].mean())

    rows = []
    for p in ALL_PLAYERS:
        with_df = base[base["_PLAYERS"].apply(lambda s: p in s)]
        without_df = base[base["_PLAYERS"].apply(lambda s: p not in s)]
        if with_df.empty or without_df.empty:
            continue
        with_net = float(with_df["CHEMISTRY_SCORE"].mean())
        without_net = float(without_df["CHEMISTRY_SCORE"].mean())
        rows.append({"PLAYER": p, "WITH_NET": with_net, "WITHOUT_NET": without_net, "DELTA": with_net - without_net})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("DELTA", ascending=False).reset_index(drop=True)
    out["WITH_NET"] = out["WITH_NET"].round(2)
    out["WITHOUT_NET"] = out["WITHOUT_NET"].round(2)
    out["DELTA"] = out["DELTA"].round(2)
    return out


def recommendation_engine(dff: pd.DataFrame, max_lineups: int = 1):
    """
    Pick "best" lineup for decision-making:
    score = (mean net) * log1p(minutes) - 0.35 * std(net)
    """
    if dff.empty:
        return None

    agg = (
        dff.groupby("GROUP_NAME", as_index=False)
        .agg(
            AVG_NET=("CHEMISTRY_SCORE", "mean"),
            STD_NET=("CHEMISTRY_SCORE", "std"),
            MIN_SUM=("MIN", "sum"),
        )
    )
    agg["STD_NET"] = agg["STD_NET"].fillna(0.0)
    agg["SCORE"] = agg["AVG_NET"] * np.log1p(agg["MIN_SUM"]) - 0.35 * agg["STD_NET"]
    pick = agg.sort_values(["SCORE", "MIN_SUM"], ascending=[False, False]).head(max_lineups).copy()
    pick["LINEUP"] = pick["GROUP_NAME"].apply(short_name)
    pick["AVG_NET"] = pick["AVG_NET"].round(2)
    pick["STD_NET"] = pick["STD_NET"].round(2)
    pick["MIN_SUM"] = pick["MIN_SUM"].round(1)
    pick["SCORE"] = pick["SCORE"].round(2)
    return pick.iloc[0].to_dict()


def worst_pairs(dff: pd.DataFrame, min_threshold: float, topn: int = 10):
    """Worst chemistry player pairs using lineup rows as co-occurrence evidence."""
    base = dff[dff["MIN"] >= (min_threshold or 0)].copy()
    if base.empty:
        return []

    pair_rows = []
    for _, r in base.iterrows():
        players = extract_players(r["GROUP_NAME"])
        if len(players) < 2:
            continue
        chem = float(r["CHEMISTRY_SCORE"])
        mins = float(r["MIN"])
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                p1, p2 = players[i], players[j]
                a, b = sorted([p1, p2])
                pair_rows.append({"PAIR": f"{a} + {b}", "CHEM": chem, "MIN": mins})

    if not pair_rows:
        return []

    pdf = pd.DataFrame(pair_rows)
    agg = (
        pdf.groupby("PAIR", as_index=False)
        .agg(
            AVG_CHEM=("CHEM", "mean"),
            MIN_SUM=("MIN", "sum"),
            SAMPLES=("PAIR", "count"),
        )
    )
    agg = agg[agg["MIN_SUM"] >= (min_threshold or 0)].copy()
    agg["AVG_CHEM"] = agg["AVG_CHEM"].round(2)
    agg["MIN_SUM"] = agg["MIN_SUM"].round(1)
    agg = agg.sort_values(["AVG_CHEM", "MIN_SUM"], ascending=[True, False]).head(topn)
    return agg.to_dict("records")


def build_synergy_graph(dff: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in dff.iterrows():
        players = extract_players(r["GROUP_NAME"])
        chem = float(r["CHEMISTRY_SCORE"])
        mins = float(r["MIN"])
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                u, v = players[i], players[j]
                if G.has_edge(u, v):
                    G[u][v]["chem_sum"] += chem
                    G[u][v]["min_sum"] += mins
                    G[u][v]["n"] += 1
                else:
                    G.add_edge(u, v, chem_sum=chem, min_sum=mins, n=1)
    return G


# ----------------------------
# Layout
# ----------------------------
def coach_hint(title: str, bullets: list[str]):
    return html.Div(
        className="card",
        children=[
            html.H4(title),
            html.Ul([html.Li(b) for b in bullets]),
        ],
        style={"marginTop": "12px"},
    )


shared_filters = html.Div(
    className="grid-2",
    children=[
        html.Div(
            className="card",
            children=[
                html.H3("Shared Controls"),
                html.Label("Minutes Threshold (applies across ALL tabs)"),
                dcc.Slider(
                    id="min-threshold",
                    min=0,
                    max=int(max(200, df["MIN"].max())),
                    step=5,
                    value=50,
                    marks={0: "0", 50: "50", 100: "100", 150: "150", 200: "200"},
                    persistence=True,
                    persistence_type="session",
                ),
                html.Div("Tip: raise this to pressure-test if chemistry holds in real minutes.", className="hint"),
            ],
        ),
        html.Div(
            className="card",
            children=[
                html.H3("Shared Player Focus"),
                html.Label("Players (select up to 5)"),
                dcc.Dropdown(
                    id="player-filter",
                    options=[{"label": p, "value": p} for p in ALL_PLAYERS],
                    multi=True,
                    placeholder="Type a name… (e.g., A. SARR)",
                    persistence=True,
                    persistence_type="session",
                ),
                html.Div(id="player-warning", className="hint"),
            ],
        ),
    ],
)


def layout_lineups():
    return html.Div(
        children=[
            html.Div(
                className="filters",
                children=[
                    html.Div(
                        className="card",
                        children=[
                            html.Label("Chemistry Tier"),
                            dcc.Checklist(
                                id="tier-filter",
                                options=[{"label": t, "value": t} for t in ["High", "Medium", "Low"]],
                                value=["High", "Medium", "Low"],
                                inline=True,
                                persistence=True,
                                persistence_type="session",
                            ),
                        ],
                    ),
                    html.Div(
                        className="card",
                        children=[
                            html.Label("Bar Chart Metric"),
                            dcc.RadioItems(
                                id="metric-mode",
                                options=[
                                    {"label": "Chemistry Score (Net Rating)", "value": "CHEMISTRY_SCORE"},
                                    {"label": "Best Chemistry per Minute", "value": "CHEM_PER_MIN"},
                                ],
                                value="CHEMISTRY_SCORE",
                                inline=False,
                                persistence=True,
                                persistence_type="session",
                            ),
                            html.Div(
                                "Tip: per-minute can overrate tiny samples — keep Minutes Threshold meaningful.",
                                className="hint",
                            ),
                        ],
                    ),
                ],
            ),
            html.Br(),
            html.Div(
                className="grid-2",
                children=[
                    html.Div(className="card", children=[html.H3("Chemistry vs Usage"), dcc.Graph(id="bubble")]),
                    html.Div(className="card", children=[html.H3(id="bar-title"), dcc.Graph(id="top-bar")]),
                ],
            ),
            coach_hint(
                "How to read this tab (coach-facing)",
                [
                    "Bubble chart: higher = better chemistry, farther right = more minutes (more trustworthy).",
                    "Bar chart: the best lineups under the Minutes Threshold and Tier filters.",
                    "Use this tab to find candidates to EXPAND/TEST/REDUCE before you watch film.",
                ],
            ),
            html.Br(),
            html.Div(
                className="grid-2",
                children=[
                    html.Div(className="card", children=[html.H3("Expand / Test / Reduce"), html.Div(id="actions-panel")]),
                    html.Div(
                        className="card",
                        children=[
                            html.H3("Top Teammates"),
                            html.Div("Based on shared lineups under current filters.", className="hint"),
                            html.Div(id="teammates-panel"),
                        ],
                    ),
                ],
            ),
        ]
    )

def layout_players():
    return html.Div(
        children=[
            # Row 1: LEFT column is a stack (dropdown + builds table), RIGHT column is fingerprint
            html.Div(
                className="grid-2",
                children=[
                    # LEFT STACK
                    html.Div(
                        style={"display": "flex", "flexDirection": "column", "gap": "14px"},
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.H3("⭐ Best 5-man Builds Around Player X"),
                                    dcc.Dropdown(
                                        id="build-player",
                                        options=[{"label": p, "value": p} for p in ALL_PLAYERS],
                                        placeholder="Pick a player…",
                                        clearable=True,
                                        persistence=True,
                                        persistence_type="session",
                                        style={"marginBottom": "8px"},
                                    ),
                                    html.Div(
                                        "Top lineups that include the player, ranked by minutes-weighted NET.",
                                        className="hint",
                                        style={"marginTop": "8px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                className="card",
                                children=[
                                    html.H3(id="best-builds-title"),
                                    html.Div(id="best-builds-panel"),
                                ],
                            ),
                        ],
                    ),

                    # RIGHT: fingerprint
                    html.Div(
                        className="card",
                        children=[
                            html.H3("Chemistry Fingerprint (Top Teammates)"),
                            html.Div("A quick snapshot of who this player pairs best with.", className="hint"),
                            dcc.Graph(id="fingerprint"),
                        ],
                    ),
                ],
            ),

            html.Br(),

            # Row 2: Coach-facing explainer next to recommendation engine
            html.Div(
                className="grid-2",
                children=[
                    html.Div(
                        className="card",
                        children=[
                            html.H3("Why this matters (coach-facing)"),
                            html.Ul(
                                [
                                    html.Li("Weighted NET = NET × Minutes so tiny samples don’t hijack the list."),
                                    html.Li("Use this as a quick menu of lineups to try when X is on the floor."),
                                    html.Li("If two lineups are close, trust the one with more minutes (more stable)."),
                                ]
                            ),
                        ],
                    ),
                    html.Div(
                        className="card",
                        children=[
                            html.H3(" Lineup Recommendation Engine (crunch-time pick)"),
                            html.Div(
                                "Picks a lineup that balances upside (NET), trust (minutes), and stability (low volatility).",
                                className="hint",
                                style={"marginBottom": "8px"},
                            ),
                            html.Div(id="recommendation-card"),
                        ],
                    ),
                ],
            ),

            html.Br(),

            # Row 3: Worst pairs full-width feel (still grid-2 so it matches site layout)
            html.Div(
                className="grid-2",
                children=[
                    html.Div(
                        className="card",
                        children=[
                            html.H3(" Who should NEVER share the floor? (Worst pairs)"),
                            html.Div(
                                "Filtered by Minutes Threshold. Use as a red flag list for film review.",
                                className="hint",
                            ),
                            html.Div(id="worst-pairs-panel"),
                        ],
                    ),
                    html.Div(
                        className="card",
                        children=[
                            html.H3("Notes (coach-facing)"),
                            html.Ul(
                                [
                                    html.Li("These are the lowest-performing pairings that still meet the minutes filter."),
                                    html.Li("Use this as a ‘check the tape’ list — not an automatic bench order."),
                                    html.Li("If a pair is unavoidable, surround them with stabilizers (high-minute, low-variance groups)."),
                                ]
                            ),
                        ],
                    ),
                ],
            ),
        ]
    )

def layout_trends():
    return html.Div(
        children=[
            html.Div(
                className="grid-2",
                children=[
                    html.Div(
                        className="card",
                        children=[
                            html.H3("Chemistry Momentum"),
                            html.Div("Rolling NET Rating over a rolling sample (ordered by minutes).", className="hint"),
                            dcc.RadioItems(
                                id="momentum-mode",
                                options=[
                                    {"label": "Focus: Selected Players (from shared filter)", "value": "players"},
                                    {"label": "Focus: Single Lineup", "value": "lineup"},
                                ],
                                value="players",
                                inline=False,
                            ),
                            dcc.Dropdown(
                                id="momentum-lineup",
                                options=[{"label": short_name(g, 80), "value": g} for g in df["GROUP_NAME"].unique()],
                                placeholder="Pick a lineup (only if Focus = Single Lineup)…",
                                clearable=True,
                            ),
                            dcc.Graph(id="momentum-graph"),
                            html.Div(id="momentum-summary", className="hint"),
                        ],
                    ),
                    html.Div(
                        className="card",
                        children=[
                            html.H3("Stability Index"),
                            html.Div("Avg NET (x) vs Volatility (y). Lower volatility = more predictable.", className="hint"),
                            dcc.Graph(id="stability-graph"),
                        ],
                    ),
                ],
            ),
            coach_hint(
                "How to read this tab (coach-facing)",
                [
                    "Momentum: is chemistry trending up or cooling off as usage grows?",
                    "Stability: separate 'high ceiling but volatile' from 'reliable winners'.",
                    "Use this tab to decide what deserves more reps vs what might be a fluke.",
                ],
            ),
            html.Br(),
            html.Div(
                className="grid-2",
                children=[
                    html.Div(
                        className="card",
                        children=[
                            html.H3("Chemistry Under Pressure"),
                            html.Div("Lineups that stayed strong in high-minute samples.", className="hint"),
                            dcc.Slider(
                                id="pressure-mins",
                                min=0,
                                max=int(max(200, df["MIN"].max())),
                                step=10,
                                value=100,
                                marks={0: "0", 100: "100", 150: "150", 200: "200"},
                            ),
                            html.Div(id="pressure-panel"),
                        ],
                    ),
                    html.Div(
                        className="card",
                        children=[
                            html.H3("Player On/Off Chemistry Lift"),
                            html.Div("Who raises the chemistry of the lineups around them?", className="hint"),
                            dcc.Graph(id="onoff-graph"),
                        ],
                    ),
                ],
            ),
        ]
    )


def layout_synergy():
    return html.Div(
        children=[
            html.Div(
                className="card",
                children=[
                    html.H3("Player Synergy Map"),
                    html.Div("Nodes = players • Line thickness = minutes together • Color = chemistry (sign)", className="sub"),
                    dcc.Graph(id="synergy-graph", style={"height": "650px"}),
                    html.Div(
                        "Tip: Use shared Player Focus to zoom the network around a few players.",
                        className="hint",
                        style={"marginTop": "8px"},
                    ),
                ],
            ),
        ]
    )


# ----------------------------
# App Layout (static tabs to avoid missing-component callback errors)
# ----------------------------
app.layout = html.Div(
    className="page",
    children=[
        html.Div(
            className="card",
            children=[
                html.H1("Wizards Chemistry Dashboard"),
                html.Div("Lineup chemistry, usage, and performance — built for decision-makers.", className="sub"),
            ],
        ),
        html.Br(),
        shared_filters,
        html.Br(),
        dcc.Tabs(
            id="tabs",
            value="tab-lineups",
            children=[
                dcc.Tab(label="Lineups", value="tab-lineups", children=layout_lineups()),
                dcc.Tab(label="Players", value="tab-players", children=layout_players()),
                dcc.Tab(label="Trends", value="tab-trends", children=layout_trends()),
                dcc.Tab(label="Synergy Map", value="tab-synergy", children=layout_synergy()),
            ],
        ),
    ],
)


# ----------------------------
# Callbacks
# ----------------------------
@app.callback(
    Output("player-filter", "value"),
    Output("player-warning", "children"),
    Input("player-filter", "value"),
)
def enforce_max_players(selected_players):
    selected_players = selected_players or []
    if len(selected_players) <= 5:
        return selected_players, ""
    trimmed = selected_players[:5]
    return trimmed, "⚠️ Max 5 players — I kept the first 5."


@app.callback(
    Output("bubble", "figure"),
    Output("top-bar", "figure"),
    Output("bar-title", "children"),
    Output("actions-panel", "children"),
    Output("teammates-panel", "children"),
    Input("min-threshold", "value"),
    Input("tier-filter", "value"),
    Input("player-filter", "value"),
    Input("metric-mode", "value"),
)
def update_lineups_tab(min_threshold, tiers, selected_players, metric_mode):
    min_threshold = min_threshold or 0
    tiers = tiers or ["High", "Medium", "Low"]
    selected_players = selected_players or []
    metric_mode = metric_mode or "CHEMISTRY_SCORE"

    dff = df[(df["MIN"] >= min_threshold) & (df["CHEM_TIER"].isin(tiers))].copy()
    if selected_players:
        dff = dff[dff["GROUP_NAME"].apply(lambda g: lineup_contains_all(g, selected_players))].copy()

    if dff.empty:
        empty_scatter = px.scatter(title="No lineups match those filters.")
        empty_scatter.update_layout(template="plotly_dark", height=520)
        empty_bar = px.bar(title="No lineups match those filters.")
        empty_bar.update_layout(template="plotly_dark", height=520)
        return empty_scatter, empty_bar, "Top Lineups", html.Div(), html.Div("No teammates to show.", className="hint")

    # Bubble
    fig_bubble = px.scatter(
        dff,
        x="MIN",
        y="CHEMISTRY_SCORE",
        size="MIN",
        color="CHEM_TIER",
        hover_name="GROUP_NAME",
        hover_data={"MIN": True, "CHEMISTRY_SCORE": True, "CHEM_PER_MIN": ":.3f", "CHEM_TIER": True},
    )
    fig_bubble.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Minutes Played",
        yaxis_title="Chemistry Score (Net Rating)",
    )

    # Bar
    metric_label = "Chemistry Score (Net Rating)" if metric_mode == "CHEMISTRY_SCORE" else "Chemistry per Minute"
    bar_title = f"Top Lineups by {metric_label}"

    top = dff.sort_values(metric_mode, ascending=False).head(10).copy()
    top["GROUP_SHORT"] = top["GROUP_NAME"].apply(short_name)

    fig_bar = px.bar(
        top.sort_values(metric_mode),
        x=metric_mode,
        y="GROUP_SHORT",
        orientation="h",
        hover_data={
            "MIN": True,
            "CHEM_TIER": True,
            "GROUP_NAME": True,
            "CHEMISTRY_SCORE": True,
            "CHEM_PER_MIN": ":.3f",
        },
    )
    fig_bar.update_yaxes(title="")
    fig_bar.update_layout(template="plotly_dark", height=520, margin=dict(l=20, r=20, t=20, b=20), xaxis_title=metric_label)

    # Actions
    expand_rows, test_rows, reduce_rows = compute_actions(dff)
    actions_children = html.Div(
        children=[
            html.H4("EXPAND (underused, high chemistry)"),
            build_table(expand_rows, ["LINEUP", "MIN", "CHEMISTRY_SCORE", "CHEM_PER_MIN", "CHEM_TIER"], page_size=6, tooltip_cols={"LINEUP"}),
            html.Br(),
            html.H4("TEST (promising per-minute, small sample)"),
            build_table(test_rows, ["LINEUP", "MIN", "CHEMISTRY_SCORE", "CHEM_PER_MIN", "CHEM_TIER"], page_size=6, tooltip_cols={"LINEUP"}),
            html.Br(),
            html.H4("REDUCE (overused, low chemistry)"),
            build_table(reduce_rows, ["LINEUP", "MIN", "CHEMISTRY_SCORE", "CHEM_PER_MIN", "CHEM_TIER"], page_size=6, tooltip_cols={"LINEUP"}),
        ]
    )

    teammates_rows = compute_top_teammates(dff, selected_players, topn=12)
    if teammates_rows:
        teammates_panel = build_table(teammates_rows, ["PLAYER", "AVG_CHEM", "LINEUP_MIN", "COUNT_LINEUPS"], page_size=8)
    else:
        teammates_panel = html.Div("Select at least 1 player to see teammates.", className="hint")

    return fig_bubble, fig_bar, bar_title, actions_children, teammates_panel


@app.callback(
    Output("best-builds-title", "children"),
    Output("best-builds-panel", "children"),
    Output("fingerprint", "figure"),
    Output("worst-pairs-panel", "children"),
    Output("recommendation-card", "children"),
    Input("build-player", "value"),
    Input("min-threshold", "value"),
    Input("tier-filter", "value"),
    Input("player-filter", "value"),
)
def update_players_tab(focus_player, min_threshold, tiers, selected_players):
    min_threshold = min_threshold or 0
    tiers = tiers or ["High", "Medium", "Low"]
    selected_players = selected_players or []

    dff = df[(df["MIN"] >= min_threshold) & (df["CHEM_TIER"].isin(tiers))].copy()
    if selected_players:
        dff = dff[dff["GROUP_NAME"].apply(lambda g: lineup_contains_all(g, selected_players))].copy()

    # Best builds table
    title = f"Best 5-man Builds Around {focus_player}" if focus_player else "Best 5-man Builds Around Player X"
    rows = compute_best_builds(df, focus_player or "", min_threshold, topn=3)

    if not focus_player:
        builds_panel = html.Div("Pick a player above to generate builds.", className="hint")
    elif not rows:
        builds_panel = html.Div("No lineups found for that player at this Minutes Threshold.", className="hint")
    else:
        builds_panel = build_table(
            rows,
            ["LINEUP", "MIN", "CHEMISTRY_SCORE", "CHEM_PER_MIN", "WEIGHTED_SCORE", "CHEM_TIER"],
            page_size=3,
            tooltip_cols={"LINEUP"},
        )

    # Fingerprint radar (top teammates by avg chem with focus player)
    fp_fig = go.Figure()
    if focus_player:
        # build teammate agg
        rows_tm = []
        base = df[df["MIN"] >= min_threshold].copy()
        base = base[base["GROUP_NAME"].apply(lambda g: focus_player in extract_players(g))].copy()
        for _, r in base.iterrows():
            ps = extract_players(r["GROUP_NAME"])
            chem = float(r["CHEMISTRY_SCORE"])
            mins = float(r["MIN"])
            for p in ps:
                if p == focus_player:
                    continue
                rows_tm.append({"TM": p, "CHEM": chem, "MIN": mins})
        if rows_tm:
            tdf = pd.DataFrame(rows_tm)
            agg = (
                tdf.groupby("TM", as_index=False)
                .agg(AVG_CHEM=("CHEM", "mean"), MIN_SUM=("MIN", "sum"))
                .sort_values(["AVG_CHEM", "MIN_SUM"], ascending=[False, False])
                .head(6)
            )
            labels = agg["TM"].tolist()
            values = agg["AVG_CHEM"].tolist()
            fp_fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill="toself", name="Avg NET"))
            fp_fig.update_layout(
                template="plotly_dark",
                showlegend=False,
                margin=dict(l=30, r=30, t=30, b=30),
                polar=dict(radialaxis=dict(visible=True)),
            )
        else:
            fp_fig.update_layout(template="plotly_dark")
    else:
        fp_fig.update_layout(template="plotly_dark")
        fp_fig.add_annotation(text="Pick a player to generate fingerprint", showarrow=False, x=0.5, y=0.5)

    # Worst pairs
    pairs_rows = worst_pairs(df, min_threshold=min_threshold, topn=10)
    if pairs_rows:
        pairs_panel = build_table(pairs_rows, ["PAIR", "AVG_CHEM", "MIN_SUM", "SAMPLES"], page_size=10)
    else:
        pairs_panel = html.Div("No pair data available at this threshold.", className="hint")

    # Recommendation engine
    pick = recommendation_engine(dff)
    if not pick:
        rec = html.Div("No lineup recommendation available under current filters.", className="hint")
    else:
        full = pick["GROUP_NAME"]
        players_list = " – ".join(extract_players(full))
        rec = html.Div(
            children=[
                html.H4("Recommended Lineup"),
                html.Div(players_list, style={"fontWeight": "700", "marginBottom": "6px"}),
                html.Div(f'Why: +{pick["AVG_NET"]} avg NET over {pick["MIN_SUM"]} minutes, volatility {pick["STD_NET"]}.'),
                html.Div("Use: close games / pressure minutes (balanced upside + stability).", className="hint"),
            ]
        )

    return title, builds_panel, fp_fig, pairs_panel, rec


@app.callback(
    Output("momentum-graph", "figure"),
    Output("momentum-summary", "children"),
    Output("stability-graph", "figure"),
    Output("pressure-panel", "children"),
    Output("onoff-graph", "figure"),
    Input("momentum-mode", "value"),
    Input("momentum-lineup", "value"),
    Input("player-filter", "value"),
    Input("min-threshold", "value"),
    Input("tier-filter", "value"),
    Input("pressure-mins", "value"),
)
def update_trends_tab(mode, lineup_name, selected_players, min_threshold, tiers, pressure_mins):
    min_threshold = min_threshold or 0
    pressure_mins = pressure_mins or 100
    tiers = tiers or ["High", "Medium", "Low"]
    selected_players = selected_players or []
    mode = mode or "players"

    dff = df[(df["MIN"] >= min_threshold) & (df["CHEM_TIER"].isin(tiers))].copy()

    # Momentum
    m = dff.copy()
    if mode == "lineup":
        if lineup_name:
            m = m[m["GROUP_NAME"] == lineup_name].copy()
        else:
            m = m.iloc[0:0].copy()
    else:
        # focus on selected players (if none, use all)
        if selected_players:
            m = m[m["GROUP_NAME"].apply(lambda g: any(p in extract_players(g) for p in selected_players))].copy()

    if m.empty:
        mom_fig = px.line(title="No momentum data for this selection.")
        mom_fig.update_layout(template="plotly_dark", height=420)
        mom_summary = "Try lowering Minutes Threshold or selecting players / a lineup."
    else:
        # rolling mean over sample idx
        m = m.sort_values("SAMPLE_IDX").copy()
        m["ROLL_NET"] = m["CHEMISTRY_SCORE"].rolling(window=10, min_periods=1).mean()
        mom_fig = px.line(m, x="SAMPLE_IDX", y="ROLL_NET", title="Rolling NET (Rolling Sample)")
        mom_fig.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=50, b=20))
        # trend
        delta = float(m["ROLL_NET"].iloc[-1] - m["ROLL_NET"].iloc[0])
        arrow = "↑" if delta > 0.5 else ("↓" if delta < -0.5 else "→")
        mom_summary = f"{arrow} Rolling NET change: {delta:+.2f} (ordered by minutes; label: Rolling Sample)."

    # Stability Index (per lineup)
    stab = stability_index(dff)
    if stab.empty:
        stab_fig = px.scatter(title="No stability data.")
        stab_fig.update_layout(template="plotly_dark", height=420)
    else:
        stab["GROUP_SHORT"] = stab["GROUP_NAME"].apply(lambda s: short_name(s, 75))
        stab_fig = px.scatter(
            stab,
            x="AVG_NET",
            y="VAR_NET",
            size="MIN_SUM",
            hover_name="GROUP_NAME",
            title="Avg NET vs Variance (Stability)",
        )
        stab_fig.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=50, b=20))
        stab_fig.update_xaxes(title="Avg NET (higher = better)")
        stab_fig.update_yaxes(title="Variance (lower = more stable)")

    # Chemistry Under Pressure
    pressure_df = dff[dff["MIN"] >= pressure_mins].copy()
    if pressure_df.empty:
        pressure_panel = html.Div("No high-minute lineups at this threshold.", className="hint")
    else:
        top = (
            pressure_df.sort_values(["CHEMISTRY_SCORE", "MIN"], ascending=[False, False])
            .head(10)[["GROUP_NAME", "MIN", "CHEMISTRY_SCORE", "CHEM_TIER"]]
            .copy()
        )
        top["LINEUP"] = top["GROUP_NAME"].apply(short_name)
        top["MIN"] = top["MIN"].round(1)
        top["CHEMISTRY_SCORE"] = top["CHEMISTRY_SCORE"].round(2)
        rows = top[["LINEUP", "MIN", "CHEMISTRY_SCORE", "CHEM_TIER"]].to_dict("records")
        pressure_panel = build_table(rows, ["LINEUP", "MIN", "CHEMISTRY_SCORE", "CHEM_TIER"], page_size=10, tooltip_cols={"LINEUP"})

    # On/Off
    lift = on_off_lift(df, min_threshold=min_threshold)
    if lift.empty:
        onoff_fig = px.bar(title="No on/off lift data.")
        onoff_fig.update_layout(template="plotly_dark", height=420)
    else:
        top_lift = lift.head(12)
        onoff_fig = px.bar(top_lift, x="DELTA", y="PLAYER", orientation="h", title="Top Chemistry Lift (With vs Without)")
        onoff_fig.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=50, b=20))
        onoff_fig.update_xaxes(title="Chemistry Lift (Avg NET with player - without)")

    return mom_fig, mom_summary, stab_fig, pressure_panel, onoff_fig


@app.callback(
    Output("synergy-graph", "figure"),
    Input("player-filter", "value"),
    Input("min-threshold", "value"),
)
def update_synergy(selected_players, min_threshold):
    selected_players = selected_players or []
    min_threshold = min_threshold or 0

    dff = df[df["MIN"] >= min_threshold].copy()
    if selected_players:
        sel = set(selected_players)
        dff = dff[dff["GROUP_NAME"].apply(lambda g: len(sel.intersection(set(extract_players(g)))) > 0)].copy()

    G = build_synergy_graph(dff)
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        fig = px.scatter(title="No synergy links available for these filters.")
        fig.update_layout(template="plotly_dark", height=650)
        return fig

    pos = nx.spring_layout(G, seed=42, k=0.7)

    # edges: use sign of avg chemistry for color (approx)
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        avg_chem = (data.get("chem_sum", 0.0) / max(1, data.get("n", 1)))
        mins = float(data.get("min_sum", 0.0))
        color = "rgba(50,255,120,0.45)" if avg_chem >= 0 else "rgba(255,80,120,0.45)"
        width = max(1.0, mins / 60.0)
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="none",
            )
        )

    node_x, node_y, node_text, node_size = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        deg = G.degree(node)
        node_size.append(10 + 2 * deg)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="bottom center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(size=node_size, color="#4da3ff", line=dict(width=1.5, color="white"), opacity=0.95),
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        template="plotly_dark",
        height=650,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
