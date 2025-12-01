# 📊 DASHBOARD FINAL: K-MEANS + APRIORI + CRUCE (Dark + Glassmorphism)

import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# ==============================
# INICIALIZACIÓN
# ==============================

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Dashboard Segmentación y Reglas de Asociación"

# ==============================
# CARGA DE DATOS
# ==============================

ruta = "./data/procesados/"

df_clusters = pd.read_csv(os.path.join(ruta, "04. clientes_clusterizados.csv"))
df_reglas_apriori = pd.read_csv(os.path.join(ruta, "03T. reglas_apriori_traducido.csv"))
df_resumen_cruce = pd.read_csv(os.path.join(ruta, "08T. resumen_reglas_apriori_clusters_traducido.csv"))

df_clusters["cluster"] = df_clusters["cluster"].astype(int)
df_resumen_cruce["cluster"] = df_resumen_cruce["cluster"].astype(int)

# ==============================
# FIGURAS BASE (NO INTERACTIVAS)
# ==============================

# Distribución clientes
fig_cluster_count = px.histogram(
    df_clusters,
    x="cluster",
    color="cluster",
    title="Distribución de clientes por clúster",
    labels={"cluster": "Clúster"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_cluster_count.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    bargap=0.25,
)

# Heatmap departamentos
cols_descartar = [
    "user_id", "n_pedidos", "orden_max",
    "dia_promedio", "hora_promedio",
    "dias_entre_pedidos", "cluster",
]
departamentos = df_clusters.drop(columns=cols_descartar, errors="ignore")
departamentos["cluster"] = df_clusters["cluster"]
heat_data = departamentos.groupby("cluster").mean().round(3)

fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=heat_data.values,
        x=heat_data.columns,
        y=heat_data.index,
        colorscale="Viridis",
        colorbar=dict(title="Consumo promedio"),
    )
)
fig_heatmap.update_layout(
    title="Consumo promedio por departamento y clúster",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

# Hora promedio por clúster
hora_prom = (
    df_clusters.groupby("cluster")["hora_promedio"]
    .mean()
    .reset_index()
)
fig_hora = px.bar(
    hora_prom,
    x="cluster",
    y="hora_promedio",
    color="cluster",
    title="Hora promedio de compra por clúster",
    labels={"hora_promedio": "Hora promedio"},
    color_discrete_sequence=px.colors.qualitative.Pastel1,
)
fig_hora.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

# Scatter global Apriori
fig_scatter_global = px.scatter(
    df_reglas_apriori,
    x="soporte",
    y="confianza",
    size="elevacion",
    color="elevacion",
    title="Reglas de asociación global: Soporte vs Confianza",
    labels={"soporte": "Soporte", "confianza": "Confianza", "elevacion": "Elevación (lift)"},
    color_continuous_scale="Viridis",
)
fig_scatter_global.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

# Red global reducida
reglas_red = df_reglas_apriori[
    (df_reglas_apriori["elevacion"] > 3.2) &
    (df_reglas_apriori["confianza"] > 0.2)
].copy()

reglas_red["antecedentes_str"] = (
    reglas_red["antecedentes"].astype(str)
    .str.strip("frozenset({})")
    .str.replace("'", "", regex=False)
)
reglas_red["consecuentes_str"] = (
    reglas_red["consecuentes"].astype(str)
    .str.strip("frozenset({})")
    .str.replace("'", "", regex=False)
)

G = nx.DiGraph()
for _, row in reglas_red.iterrows():
    ants = [p.strip() for p in row["antecedentes_str"].split(",") if p.strip()]
    cons = [p.strip() for p in row["consecuentes_str"].split(",") if p.strip()]
    for a in ants:
        for c in cons:
            G.add_edge(a, c, weight=row["elevacion"])

pos = nx.spring_layout(G, seed=42, k=0.7)
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
    line=dict(width=0.5, color="#555"),
    hoverinfo="none",
)

node_x, node_y, node_text = [], [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    text=node_text,
    textposition="top center",
    hoverinfo="text",
    marker=dict(
        size=12,
        color="#00eaff",
        line_width=1,
        line_color="#ffffff",
    ),
)

fig_red = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title="Red global de reglas de asociación (filtrada)",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=20, r=20, b=20, t=50),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark",
    ),
)

# ==============================
# UTILIDADES PARA COMENTARIOS
# ==============================

descripciones_cluster = {
    "all": "Se muestran métricas agregadas considerando todos los clústeres de clientes.",
    0: "El clúster 0 refleja un comportamiento medio en frecuencia y monto: compradores regulares, de perfil estándar.",
    1: "El clúster 1 agrupa clientes de baja actividad: pocas compras y tickets más bien acotados.",
    2: "El clúster 2 concentra compradores de alto valor: mayor gasto total y mayor intensidad de compra.",
    3: "El clúster 3 recoge clientes consistentes, con patrones estables y montos relevantes por pedido.",
    4: "El clúster 4 representa un nicho muy homogéneo, con hábitos muy repetitivos y altamente predecibles.",
}

# ==============================
# LAYOUTS POR PÁGINA
# ==============================

def home_layout():
    return html.Div(
        className="page home-page",
        children=[
            html.Div(
                className="glass-card hero-card",
                children=[
                    html.H1("Segmentación de Clientes y Patrones de Compra", className="hero-title"),
                    html.P(
                        "Dashboard interactivo para analizar clústeres de clientes, reglas de asociación y el cruce entre ambos enfoques.",
                        className="hero-subtitle",
                    ),
                ],
            ),
            html.Div(
                className="cards-grid",
                children=[
                    dcc.Link(
                        href="/kmeans",
                        className="nav-card-link",
                        children=html.Div(
                            className="glass-card nav-card",
                            children=[
                                html.H2("Segmentación K-Means", className="card-title"),
                                html.P(
                                    "Distribución de clientes, comportamiento temporal y consumo por departamento.",
                                    className="card-text",
                                ),
                            ],
                        ),
                    ),
                    dcc.Link(
                        href="/apriori",
                        className="nav-card-link",
                        children=html.Div(
                            className="glass-card nav-card",
                            children=[
                                html.H2("Reglas de Asociación (Apriori)", className="card-title"),
                                html.P(
                                    "Exploración de soporte, confianza, elevación y red de productos asociados.",
                                    className="card-text",
                                ),
                            ],
                        ),
                    ),
                    dcc.Link(
                        href="/cruce",
                        className="nav-card-link",
                        children=html.Div(
                            className="glass-card nav-card",
                            children=[
                                html.H2("Cruce Apriori + Clústeres", className="card-title"),
                                html.P(
                                    "Reglas por segmento de clientes y comparación de patrones entre clústeres.",
                                    className="card-text",
                                ),
                            ],
                        ),
                    ),
                ],
            ),
        ],
    )


def kmeans_layout():
    clusters_unicos = sorted(df_clusters["cluster"].unique())
    options = [{"label": "Todos", "value": "all"}] + [
        {"label": f"Clúster {c}", "value": c} for c in clusters_unicos
    ]

    return html.Div(
        className="page content-page",
        children=[
            html.Div(
                className="page-header",
                children=[
                    html.H1("Segmentación de clientes – K-Means", className="page-title"),
                    html.P(
                        "Análisis de distribución de clientes, comportamiento temporal y consumo por categoría.",
                        className="page-subtitle",
                    ),
                ],
            ),
            html.Div(
                className="filters-row glass-card",
                children=[
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Seleccionar clúster", className="filter-label"),
                            dcc.Dropdown(
                                id="kmeans-cluster-dropdown",
                                options=options,
                                value="all",
                                clearable=False,
                                className="dropdown-dark",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="graphs-grid",
                children=[
                    html.Div(
                        className="glass-card graph-card",
                        children=[
                            dcc.Graph(id="kmeans-metricas-cluster"),
                            html.Div(id="kmeans-comentario", className="graph-comment"),
                        ],
                    ),
                    html.Div(
                        className="glass-card graph-card",
                        children=[
                            dcc.Graph(figure=fig_cluster_count),
                        ],
                    ),
                    html.Div(
                        className="glass-card graph-card wide",
                        children=[
                            dcc.Graph(figure=fig_hora),
                        ],
                    ),
                    html.Div(
                        className="glass-card graph-card wide",
                        children=[
                            dcc.Graph(figure=fig_heatmap),
                        ],
                    ),
                ],
            ),
        ],
    )


def apriori_layout():
    soporte_min = float(df_reglas_apriori["soporte"].min())
    soporte_max = float(df_reglas_apriori["soporte"].max())
    confianza_min = float(df_reglas_apriori["confianza"].min())
    confianza_max = float(df_reglas_apriori["confianza"].max())
    elev_min = float(df_reglas_apriori["elevacion"].min())
    elev_max = float(df_reglas_apriori["elevacion"].max())

    return html.Div(
        className="page content-page",
        children=[
            html.Div(
                className="page-header",
                children=[
                    html.H1("Reglas de Asociación – Apriori", className="page-title"),
                    html.P(
                        "Exploración global de reglas según soporte, confianza y elevación, complementada con una red de productos.",
                        className="page-subtitle",
                    ),
                ],
            ),
            html.Div(
                className="filters-row glass-card",
                children=[
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Soporte mínimo", className="filter-label"),
                            dcc.Slider(
                                id="apriori-slider-soporte",
                                min=soporte_min,
                                max=soporte_max,
                                step=(soporte_max - soporte_min) / 50,
                                value=soporte_min,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Confianza mínima", className="filter-label"),
                            dcc.Slider(
                                id="apriori-slider-confianza",
                                min=confianza_min,
                                max=confianza_max,
                                step=(confianza_max - confianza_min) / 50,
                                value=confianza_min,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Elevación mínima (lift)", className="filter-label"),
                            dcc.Slider(
                                id="apriori-slider-elevacion",
                                min=elev_min,
                                max=elev_max,
                                step=(elev_max - elev_min) / 50,
                                value=elev_min,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="graphs-grid",
                children=[
                    html.Div(
                        className="glass-card graph-card wide",
                        children=[
                            dcc.Graph(id="apriori-scatter-filtrado"),
                            html.Div(id="apriori-info", className="graph-comment"),
                        ],
                    ),
                    html.Div(
                        className="glass-card graph-card wide",
                        children=[
                            dcc.Graph(figure=fig_red),
                        ],
                    ),
                ],
            ),
        ],
    )


def cruce_layout():
    clusters_unicos = sorted(df_resumen_cruce["cluster"].unique())
    options = [{"label": f"Clúster {c}", "value": c} for c in clusters_unicos]

    return html.Div(
        className="page content-page",
        children=[
            html.Div(
                className="page-header",
                children=[
                    html.H1("Cruce Apriori + K-Means", className="page-title"),
                    html.P(
                        "Visualización de reglas de asociación segmentadas por clúster, para comparar patrones entre grupos de clientes.",
                        className="page-subtitle",
                    ),
                ],
            ),
            html.Div(
                className="filters-row glass-card",
                children=[
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Clústeres", className="filter-label"),
                            dcc.Dropdown(
                                id="cruce-cluster-dropdown",
                                options=options,
                                value=[c for c in clusters_unicos],
                                multi=True,
                                className="dropdown-dark",
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Elevación mínima (lift)", className="filter-label"),
                            dcc.Slider(
                                id="cruce-slider-elevacion",
                                min=float(df_resumen_cruce["elevacion"].min()),
                                max=float(df_resumen_cruce["elevacion"].max()),
                                step=float(
                                    (df_resumen_cruce["elevacion"].max() - df_resumen_cruce["elevacion"].min()) / 50
                                ),
                                value=float(df_resumen_cruce["elevacion"].min()),
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="graphs-grid",
                children=[
                    html.Div(
                        className="glass-card graph-card wide",
                        children=[
                            dcc.Graph(id="cruce-scatter"),
                            html.Div(id="cruce-info", className="graph-comment"),
                        ],
                    ),
                ],
            ),
        ],
    )

# ==============================
# SIDEBAR + LAYOUT GENERAL
# ==============================

sidebar = html.Div(
    id="sidebar",
    className="sidebar expanded",
    children=[
        html.Div(
            className="sidebar-header",
            children=[
                html.Div("PT", className="sidebar-logo"),
                html.H2("Proyecto de Título", className="sidebar-title"),
            ],
        ),
        html.Div(
            className="sidebar-menu",
            children=[
                dcc.Link("Inicio", href="/", className="sidebar-link"),
                dcc.Link("K-Means", href="/kmeans", className="sidebar-link"),
                dcc.Link("Apriori", href="/apriori", className="sidebar-link"),
                dcc.Link("Cruce", href="/cruce", className="sidebar-link"),
            ],
        ),
    ],
)

app.layout = html.Div(
    className="app-container",
    children=[
        dcc.Location(id="url"),
        html.Button("☰", id="btn-sidebar", className="sidebar-toggle"),
        sidebar,
        html.Div(id="page-content", className="content-wrapper"),
    ],
)

# ==============================
# CALLBACKS
# ==============================

# Toggle sidebar
@app.callback(
    Output("sidebar", "className"),
    Input("btn-sidebar", "n_clicks"),
    prevent_initial_call=False,
)
def toggle_sidebar(n_clicks):
    if not n_clicks or n_clicks % 2 == 0:
        return "sidebar expanded"
    else:
        return "sidebar collapsed"


# Navegación de páginas
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    if pathname == "/kmeans":
        return kmeans_layout()
    elif pathname == "/apriori":
        return apriori_layout()
    elif pathname == "/cruce":
        return cruce_layout()
    else:
        return home_layout()


# Métricas de K-Means por clúster
@app.callback(
    Output("kmeans-metricas-cluster", "figure"),
    Output("kmeans-comentario", "children"),
    Input("kmeans-cluster-dropdown", "value"),
)
def actualizar_kmeans_cluster(cluster_sel):
    if cluster_sel == "all":
        df_sel = df_clusters.copy()
    else:
        df_sel = df_clusters[df_clusters["cluster"] == cluster_sel].copy()

    resumen = df_sel[["n_pedidos", "orden_max", "dias_entre_pedidos"]].mean().round(2)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Número de pedidos", "Monto máximo orden", "Días entre pedidos"],
            y=[resumen["n_pedidos"], resumen["orden_max"], resumen["dias_entre_pedidos"]],
            marker_color=["#00eaff", "#9d4edd", "#7effa5"],
        )
    )
    fig.update_layout(
        title="Métricas promedio del clúster seleccionado",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    comentario = descripciones_cluster.get(cluster_sel, descripciones_cluster["all"])
    return fig, comentario


# Scatter de Apriori filtrado
@app.callback(
    Output("apriori-scatter-filtrado", "figure"),
    Output("apriori-info", "children"),
    Input("apriori-slider-soporte", "value"),
    Input("apriori-slider-confianza", "value"),
    Input("apriori-slider-elevacion", "value"),
)
def filtrar_reglas_apriori(soporte_min, confianza_min, elev_min):
    df_filtrado = df_reglas_apriori[
        (df_reglas_apriori["soporte"] >= soporte_min)
        & (df_reglas_apriori["confianza"] >= confianza_min)
        & (df_reglas_apriori["elevacion"] >= elev_min)
    ]

    fig = px.scatter(
        df_filtrado,
        x="soporte",
        y="confianza",
        size="elevacion",
        color="elevacion",
        title="Reglas filtradas según soporte, confianza y elevación",
        labels={"soporte": "Soporte", "confianza": "Confianza", "elevacion": "Elevación (lift)"},
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    texto = (
        f"Se muestran {len(df_filtrado)} reglas que cumplen: "
        f"soporte ≥ {soporte_min:.3f}, confianza ≥ {confianza_min:.3f}, elevación ≥ {elev_min:.3f}."
    )
    return fig, texto


# Scatter del cruce Apriori + K-Means
@app.callback(
    Output("cruce-scatter", "figure"),
    Output("cruce-info", "children"),
    Input("cruce-cluster-dropdown", "value"),
    Input("cruce-slider-elevacion", "value"),
)
def filtrar_cruce(cluster_list, elev_min):
    if not cluster_list:
        df_filtrado = df_resumen_cruce[df_resumen_cruce["elevacion"] >= elev_min]
    else:
        df_filtrado = df_resumen_cruce[
            (df_resumen_cruce["cluster"].isin(cluster_list))
            & (df_resumen_cruce["elevacion"] >= elev_min)
        ]

    fig = px.scatter(
        df_filtrado,
        x="confianza",
        y="elevacion",
        size="soporte",
        color="cluster",
        title="Reglas por clúster según confianza y elevación",
        labels={"confianza": "Confianza", "elevacion": "Elevación (lift)", "cluster": "Clúster"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    texto = (
        f"Se muestran {len(df_filtrado)} reglas con elevación ≥ {elev_min:.3f}, "
        f"para los clústeres seleccionados."
    )
    return fig, texto


# ==============================
# EJECUCIÓN
# ==============================

if __name__ == "__main__":
    app.run(debug=True)
