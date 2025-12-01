# pages/apriori.py
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import dash
from dash import html, dcc, Input, Output, callback

dash.register_page(
    __name__,
    path="/apriori",
    name="Apriori",
    title="Apriori – Proyecto de Título",
)

RUTA = "./data/procesados/"
df_rules = pd.read_csv(os.path.join(RUTA, "03T. reglas_apriori_traducido.csv"))

# Limpieza básica
for col in ["soporte", "confianza", "elevacion"]:
    if col in df_rules.columns:
        df_rules[col] = pd.to_numeric(df_rules[col], errors="coerce")

df_rules = df_rules.dropna(subset=["soporte", "confianza", "elevacion"])

# Productos únicos para filtro
productos_unicos = sorted(
    set(df_rules["antecedentes"].astype(str).unique())
    | set(df_rules["consecuentes"].astype(str).unique())
)


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
        html.H2("Apriori – Reglas de Asociación", className="page-title"),
        html.P(
            "Explora las reglas de asociación globales: distribución soporte-confianza, "
            "red de productos, top reglas y matriz de elevación.",
            className="page-subtitle",
        ),

        # Filtros comunes
        html.Div(
            className="glass-card",
            children=[
                html.Div("Filtros globales", className="card-title"),
                html.Div(
                    className="filter-panel",
                    children=[
                        html.Div(
                            className="filter-field",
                            children=[
                                html.Div("Elevación mínima (lift)", className="filter-label"),
                                dcc.Slider(
                                    id="apr-min-lift",
                                    min=float(df_rules["elevacion"].min()),
                                    max=float(df_rules["elevacion"].max()),
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
                                    id="apr-min-conf",
                                    min=float(df_rules["confianza"].min()),
                                    max=float(df_rules["confianza"].max()),
                                    step=0.01,
                                    value=0.2,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ],
                        ),
                        html.Div(
                            className="filter-field",
                            children=[
                                html.Div("Producto (opcional)", className="filter-label"),
                                dcc.Dropdown(
                                    id="apr-producto",
                                    options=[{"label": p, "value": p} for p in productos_unicos],
                                    placeholder="Filtrar reglas que involucren un producto...",
                                    multi=False,
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    "Estos filtros afectan a todos los gráficos de esta sección.",
                    className="helper-text",
                ),
            ],
        ),

        html.Br(),

        # Grid de gráficos (scatter + red completa)
        html.Div(
            className="card-grid-2",
            children=[
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div(
                            "01. Reglas de Asociación – Soporte vs Confianza",
                            className="card-title",
                        ),
                        html.Div(
                            "Dispersión global de reglas, donde el tamaño y el color reflejan la elevación.",
                            className="card-subtitle",
                        ),
                        dcc.Graph(id="apr-fig-scatter", className="graph-full"),
                    ],
                ),
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div(
                            "02. Red de Reglas – Vista Amplia",
                            className="card-title",
                        ),
                        html.Div(
                            "Red de productos conectados por reglas con alta elevación.",
                            className="card-subtitle",
                        ),
                        dcc.Graph(id="apr-fig-network-wide", className="graph-full"),
                    ],
                ),
            ],
        ),

        html.Br(),

        # Grid barras + heatmap + red reducida
        html.Div(
            className="card-grid-2",
            children=[
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div(
                            "03. Top reglas por elevación",
                            className="card-title",
                        ),
                        html.Div(
                            className="filter-field",
                            children=[
                                html.Div("Número máximo de reglas", className="filter-label"),
                                dcc.Slider(
                                    id="apr-top-n",
                                    min=5,
                                    max=30,
                                    step=1,
                                    value=10,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ],
                        ),
                        dcc.Graph(id="apr-fig-bar", className="graph-full"),
                    ],
                ),
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div(
                            "04. Matriz de calor de elevación (antecedente vs consecuente)",
                            className="card-title",
                        ),
                        html.Div(
                            className="filter-field",
                            children=[
                                html.Div("Número máximo de productos", className="filter-label"),
                                dcc.Slider(
                                    id="apr-max-prod",
                                    min=4,
                                    max=12,
                                    step=1,
                                    value=6,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ],
                        ),
                        dcc.Graph(id="apr-fig-heatmap", className="graph-full"),
                    ],
                ),
            ],
        ),

        html.Br(),

        html.Div(
            className="glass-card",
            children=[
                html.Div(
                    "05. Red de Reglas – Vista Reducida (Top nodos por centralidad)",
                    className="card-title",
                ),
                dcc.Graph(id="apr-fig-network-small", className="graph-full"),
            ],
        ),
    ]
)


def filtrar_reglas(min_lift, min_conf, producto):
    df = df_rules.copy()
    df = df[(df["elevacion"] >= min_lift) & (df["confianza"] >= min_conf)]
    if producto:
        df = df[
            df["antecedentes"].astype(str).str.contains(producto)
            | df["consecuentes"].astype(str).str.contains(producto)
        ]
    return df


@callback(
    Output("apr-fig-scatter", "figure"),
    Input("apr-min-lift", "value"),
    Input("apr-min-conf", "value"),
    Input("apr-producto", "value"),
)
def actualizar_scatter(min_lift, min_conf, producto):
    df = filtrar_reglas(min_lift, min_conf, producto)
    fig = px.scatter(
        df,
        x="soporte",
        y="confianza",
        size="elevacion",
        color="elevacion",
        hover_data=["antecedentes", "consecuentes"],
        color_continuous_scale="Viridis",
        labels={"soporte": "Soporte", "confianza": "Confianza", "elevacion": "Elevación (lift)"},
        title="Reglas de Asociación: Soporte vs Confianza",
    )
    return style_fig(fig)


@callback(
    Output("apr-fig-bar", "figure"),
    Input("apr-min-lift", "value"),
    Input("apr-min-conf", "value"),
    Input("apr-producto", "value"),
    Input("apr-top-n", "value"),
)
def actualizar_barras(min_lift, min_conf, producto, top_n):
    df = filtrar_reglas(min_lift, min_conf, producto)
    df = df.sort_values("elevacion", ascending=False).head(int(top_n))
    df["regla"] = df["antecedentes"] + " → " + df["consecuentes"]
    fig = px.bar(
        df,
        x="elevacion",
        y="regla",
        orientation="h",
        text="elevacion",
        labels={"elevacion": "Elevación (lift)", "regla": "Regla de asociación"},
        title=f"Top {len(df)} reglas por elevación",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis=dict(automargin=True))
    return style_fig(fig)


@callback(
    Output("apr-fig-heatmap", "figure"),
    Input("apr-min-lift", "value"),
    Input("apr-min-conf", "value"),
    Input("apr-producto", "value"),
    Input("apr-max-prod", "value"),
)
def actualizar_heatmap(min_lift, min_conf, producto, max_prod):
    df = filtrar_reglas(min_lift, min_conf, producto)
    # Tomamos los productos más frecuentes
    top_ants = (
        df["antecedentes"].value_counts().head(int(max_prod)).index.tolist()
    )
    top_cons = (
        df["consecuentes"].value_counts().head(int(max_prod)).index.tolist()
    )
    df_f = df[df["antecedentes"].isin(top_ants) & df["consecuentes"].isin(top_cons)]
    if df_f.empty:
        fig = go.Figure()
        fig.update_layout(title="No hay reglas con los filtros seleccionados.")
        return style_fig(fig)

    tabla = (
        df_f.pivot_table(
            index="antecedentes",
            columns="consecuentes",
            values="elevacion",
            aggfunc="max",
        )
        .reindex(index=top_ants, columns=top_cons)
        .fillna(0)
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=tabla.values,
            x=tabla.columns,
            y=tabla.index,
            colorscale="Turbo",
            colorbar_title="Lift",
        )
    )
    fig.update_layout(
        title="Matriz de calor de elevación (antecedente vs consecuente)",
        xaxis_title="Producto consecuente",
        yaxis_title="Producto antecedente",
    )
    return style_fig(fig)


def construir_red(df, max_nodos=None):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        a = row["antecedentes"]
        c = row["consecuentes"]
        lift = row["elevacion"]
        G.add_node(a)
        G.add_node(c)
        G.add_edge(a, c, weight=lift)

    if max_nodos and len(G.nodes) > max_nodos:
        # seleccionamos nodos por degree
        grados = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodos]
        nodos_kept = set(n for n, _ in grados)
        G = G.subgraph(nodos_kept).copy()

    if not G.nodes:
        return go.Figure()

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
            color="#38bdf8",
            line=dict(width=1, color="#0f172a"),
        ),
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )
    return style_fig(fig)


@callback(
    Output("apr-fig-network-wide", "figure"),
    Input("apr-min-lift", "value"),
    Input("apr-min-conf", "value"),
    Input("apr-producto", "value"),
)
def actualizar_red_amplia(min_lift, min_conf, producto):
    df = filtrar_reglas(min_lift, min_conf, producto).copy()
    # dejamos hasta 150 reglas para que sea manejable
    df = df.sort_values("elevacion", ascending=False).head(150)
    fig = construir_red(df, max_nodos=None)
    fig.update_layout(title="Red de reglas – vista amplia")
    return fig


@callback(
    Output("apr-fig-network-small", "figure"),
    Input("apr-min-lift", "value"),
    Input("apr-min-conf", "value"),
    Input("apr-producto", "value"),
)
def actualizar_red_reducida(min_lift, min_conf, producto):
    df = filtrar_reglas(min_lift, min_conf, producto).copy()
    df = df.sort_values("elevacion", ascending=False).head(60)
    fig = construir_red(df, max_nodos=25)
    fig.update_layout(title="Red de reglas – vista reducida (nodos más centrales)")
    return fig
