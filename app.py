from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px

DATA_PATH = "data/wizards_chemistry.csv"

app = Dash(__name__)
app.title = "Wizards Chemistry Dashboard"

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Basic safety: normalize column names if needed
    df.columns = [c.strip().upper() for c in df.columns]

    required = {"GROUP_NAME", "MIN", "CHEMISTRY_SCORE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. You have: {list(df.columns)}")

    # Chemistry tiers (simple + effective)
    df["CHEM_TIER"] = pd.cut(
        df["CHEMISTRY_SCORE"],
        bins=[-10, -0.25, 0.25, 10],
        labels=["Low", "Medium", "High"]
    )

    return df

df = load_data()

app.layout = html.Div(
    className="page",
    children=[
        # ===== HEADER =====
        html.Div(
            className="card",
            children=[
                html.H1("Wizards Chemistry Dashboard"),
                html.Div(
                    "Lineup chemistry, usage, and performance — built for decision-makers.",
                    className="sub",
                ),
            ],
        ),

        html.Br(),

        dcc.Tabs(
            value="tab-lineups",
            children=[
                dcc.Tab(label="Lineups", value="tab-lineups"),
                dcc.Tab(label="Players (later)", value="tab-players"),
                dcc.Tab(label="Trends (later)", value="tab-trends"),
            ],
        ),

        html.Br(),

        # ===== FILTERS (2 cards side-by-side) =====
        html.Div(
            className="filters",
            children=[
                # Minutes slider card
                html.Div(
                    className="card",
                    children=[
                        html.Label("Minutes Threshold (keep lineups with MIN >= this)"),
                        dcc.Slider(
                            id="min-threshold",
                            min=0,
                            max=int(df["MIN"].max()),
                            step=5,
                            value=50,
                            marks={0: "0", 50: "50", 100: "100", 150: "150", 200: "200"},
                        ),
                    ],
                ),

                # Tier checklist card
                html.Div(
                    className="card",
                    children=[
                        html.Label("Chemistry Tier"),
                        dcc.Checklist(
                            id="tier-filter",
                            options=[{"label": t, "value": t} for t in ["High", "Medium", "Low"]],
                            value=["High", "Medium", "Low"],
                            inline=True,
                        ),
                    ],
                ),
            ],
        ),

        html.Br(),

        # ===== CHARTS GRID =====
        html.Div(
            className="grid-2",
            children=[
                html.Div(
                    className="card",
                    children=[
                        html.H3("Chemistry vs Usage (Bubble)"),
                        dcc.Graph(id="bubble", className="graph"),
                    ],
                ),
                html.Div(
                    className="card",
                    children=[
                        html.H3("Top Lineups by Chemistry"),
                        dcc.Graph(id="top-bar", className="graph"),
                    ],
                ),
            ],
        ),
    ],
)

@app.callback(
    Output("bubble", "figure"),
    Output("top-bar", "figure"),
    Input("min-threshold", "value"),
    Input("tier-filter", "value"),
)
def update_charts(min_threshold, tiers):
    dff = df[(df["MIN"] >= min_threshold) & (df["CHEM_TIER"].isin(tiers))].copy()

    # Bubble chart: X=MIN, Y=CHEMISTRY_SCORE, size=MIN, color=CHEM_TIER
    fig_bubble = px.scatter(
        dff,
        x="MIN",
        y="CHEMISTRY_SCORE",
        size="MIN",
        color="CHEM_TIER",
        hover_name="GROUP_NAME",
        hover_data={"MIN": True, "CHEMISTRY_SCORE": True, "CHEM_TIER": True},
    )
    fig_bubble.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        height=520,
    )

    # Quadrant lines (vertical = threshold, horizontal = average chemistry of filtered set)
    avg_chem = float(dff["CHEMISTRY_SCORE"].mean()) if len(dff) else 0.0
    fig_bubble.add_vline(x=min_threshold, line_dash="dash", line_width=2)
    fig_bubble.add_hline(y=avg_chem, line_dash="dash", line_width=2)

    # Label the threshold line
    fig_bubble.add_annotation(
        x=min_threshold,
        y=dff["CHEMISTRY_SCORE"].max() if len(dff) else 0,
        text=f"{min_threshold} Min Threshold",
        showarrow=False,
        yshift=10
    )

    # Bar chart: top 10 by chemistry
    top = dff.sort_values("CHEMISTRY_SCORE", ascending=False).head(10)
    fig_bar = px.bar(
        top.sort_values("CHEMISTRY_SCORE"),
        x="CHEMISTRY_SCORE",
        y="GROUP_NAME",
        orientation="h",
        hover_data={"MIN": True, "CHEM_TIER": True},
    )
    fig_bar.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        height=520,
    )

    return fig_bubble, fig_bar

if __name__ == "__main__":
    app.run(debug=True)
