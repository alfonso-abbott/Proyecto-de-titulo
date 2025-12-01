# pages/cruce.py
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

import dash
from dash import html, dcc, Input, Output, callback

dash.register_page(
    __name__,
    path="/cruce",
    name="Cruce",
    title="Cruce Apriori + K-Means – Proyecto de Título",
)

RUTA = "./data/procesados/"
df_cruce = pd.read_csv(os.path.join(RUTA, "08T. resumen_reglas_apriori_clusters_traducido.csv"))

# Aseguramos nombres / tipos
cluster_col = "cluster" if "cluster" in df_cruce.columns else "Cluster"
df_cruce[cluster_col] = df_cruce[cluster_col].astype(int)

for col in ["soporte", "confianza", "elevacion"]:
    if col in df_cruce.columns:
        df_cruce[col] = pd.to_numeric(df_cruce[col], errors="coerce")

df_cruce = df_cruce.dropna(subset=["soporte", "confianza", "elevacion"])

df_cruce["regla"] = df_cruce["antecedentes"] + " → " + df_cruce["consecuentes"]

clusters_opciones = sorted(df_cruce[cluster_col].unique())


def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=10, t=40, b=40),
        legend_title_text="",
    )
    return fig


layout = html.Div(
    children=[
        html.H2("Cruce Apriori + K-Means", className="page-title"),
        html.P(
            "Reglas de asociación segmentadas por clúster para identificar patrones específicos de cada grupo.",
            className="page-subtitle",
        ),

        html.Div(
            className="glass-card",
            children=[
                html.Div("Filtros de reglas segmentadas", className="card-title"),
                html.Div(
                    className="filter-panel",
                    children=[
                        html.Div(
                            className="filter-field",
                            children=[
                                html.Div("Clúster", className="filter-label"),
                                dcc.Dropdown(
                                    id="cr-cluster",
                                    options=[
                                        {"label": f"Clúster {c}", "value": int(c)}
                                        for c in clusters_opciones
                                    ],
                                    value=int(clusters_opciones[0]),
                                    clearable=False,
                                ),
                            ],
                        ),
                        html.Div(
                            className="filter-field",
                            children=[
                                html.Div("Elevación mínima (lift)", className="filter-label"),
                                dcc.Slider(
                                    id="cr-min-lift",
                                    min=float(df_cruce["elevacion"].min()),
                                    max=float(df_cruce["elevacion"].max()),
                                    step=0.1,
                                    value=2.0,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ],
                        ),
                        html.Div(
                            className="filter-field",
                            children=[
                                html.Div("Confianza mínima", className="filter-label"),
                                dcc.Slider(
                                    id="cr-min-conf",
                                    min=float(df_cruce["confianza"].min()),
                                    max=float(df_cruce["confianza"].max()),
                                    step=0.01,
                                    value=0.2,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    "Los filtros aplican tanto al gráfico de barras como a la red del clúster seleccionado.",
                    className="helper-text",
                ),
            ],
        ),

        html.Br(),

        html.Div(
            className="card-grid-2",
            children=[
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div(
                            "01. Top reglas por clúster",
                            className="card-title",
                        ),
                        html.Div(
                            className="filter-field",
                            children=[
                                html.Div("Número máximo de reglas por clúster", className="filter-label"),
                                dcc.Slider(
                                    id="cr-top-n",
                                    min=3,
                                    max=20,
                                    step=1,
                                    value=5,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ],
                        ),
                        dcc.Graph(id="cr-fig-top", className="graph-full"),
                    ],
                ),
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div(
                            "02. Red de reglas para el clúster seleccionado",
                            className="card-title",
                        ),
                        dcc.Graph(id="cr-fig-net", className="graph-full"),
                    ],
                ),
            ],
        ),
    ]
)


def filtrar(df, cl, min_lift, min_conf):
    d = df[df[cluster_col] == cl].copy()
    d = d[(d["elevacion"] >= min_lift) & (d["confianza"] >= min_conf)]
    return d


@callback(
    Output("cr-fig-top", "figure"),
    Input("cr-cluster", "value"),
    Input("cr-min-lift", "value"),
    Input("cr-min-conf", "value"),
    Input("cr-top-n", "value"),
)
def fig_top(cluster_sel, min_lift, min_conf, top_n):
    d = filtrar(df_cruce, cluster_sel, min_lift, min_conf)
    d = d.sort_values("elevacion", ascending=False).head(int(top_n))
    if d.empty:
        fig = go.Figure()
        fig.update_layout(title="No hay reglas que cumplan los filtros.")
        return style_fig(fig)

    fig = px.bar(
        d,
        x="elevacion",
        y="regla",
        orientation="h",
        text="elevacion",
        labels={"elevacion": "Elevación (lift)", "regla": "Regla de asociación"},
        color=cluster_col,
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        title=f"Top {len(d)} reglas para el clúster {cluster_sel}",
        yaxis=dict(automargin=True),
    )
    return style_fig(fig)


def construir_red_cluster(df_cluster):
    G = nx.DiGraph()
    for _, row in df_cluster.iterrows():
        a = row["antecedentes"]
        c = row["consecuentes"]
        lift = row["elevacion"]
        G.add_node(a)
        G.add_node(c)
        G.add_edge(a, c, weight=lift)

    if not G.nodes:
        fig = go.Figure()
        fig.update_layout(title="Sin nodos para mostrar.")
        return style_fig(fig)

    # limitamos nodos para que no explote
    if len(G.nodes) > 40:
        grados = sorted(G.degree, key=lambda x: x[1], reverse=True)[:40]
        nodos_kept = set(n for n, _ in grados)
        G = G.subgraph(nodos_kept).copy()

    pos = nx.spring_layout(G, seed=42, k=0.6)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.6, color="#6b7280"),
        hoverinfo="none",
    )

    node_x, node_y, text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=text,
        textposition="top center",
        marker=dict(
            size=12,
            color="#f97316",
            line=dict(width=1, color="#0f172a"),
        ),
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        title="Red reducida de reglas para el clúster",
    )
    return style_fig(fig)


@callback(
    Output("cr-fig-net", "figure"),
    Input("cr-cluster", "value"),
    Input("cr-min-lift", "value"),
    Input("cr-min-conf", "value"),
)
def fig_net(cluster_sel, min_lift, min_conf):
    d = filtrar(df_cruce, cluster_sel, min_lift, min_conf)
    return construir_red_cluster(d)
