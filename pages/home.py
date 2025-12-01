# pages/home.py
import os
import pandas as pd
import plotly.express as px
from dash import html, dcc
import dash

dash.register_page(__name__, path="/", name="Inicio")

RUTA = "./data/procesados/"

# Figuras pequeñas de preview
df_apriori = pd.read_csv(os.path.join(RUTA, "03T. reglas_apriori_traducido.csv"))
df_clusters = pd.read_csv(os.path.join(RUTA, "04. clientes_clusterizados.csv"))
df_cruce = pd.read_csv(os.path.join(RUTA, "08T. resumen_reglas_apriori_clusters_traducido.csv"))

for col in ["soporte", "confianza", "elevacion"]:
    if col in df_apriori.columns:
        df_apriori[col] = pd.to_numeric(df_apriori[col], errors="coerce")

if "cluster" in df_clusters.columns:
    df_clusters["cluster"] = df_clusters["cluster"].astype(int)

if "cluster" in df_cruce.columns:
    df_cruce["cluster"] = df_cruce["cluster"].astype(int)

# Preview Apriori: scatter simple
fig_preview_apriori = px.scatter(
    df_apriori.sample(min(200, len(df_apriori))), 
    x="soporte", y="confianza", color="elevacion",
    color_continuous_scale="Viridis"
)
fig_preview_apriori.update_layout(
    template="plotly_dark",
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=False
)

# Preview Kmeans: hist de clusters
fig_preview_kmeans = px.histogram(
    df_clusters, x="cluster", nbins=5, color="cluster"
)
fig_preview_kmeans.update_layout(
    template="plotly_dark",
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=False
)

# Preview Cruce: top reglas por cluster
if "elevacion" in df_cruce.columns:
    df_cruce_top = (df_cruce.sort_values("elevacion", ascending=False)
                    .head(80))
    fig_preview_cruce = px.scatter(
        df_cruce_top,
        x="confianza",
        y="elevacion",
        color="cluster" if "cluster" in df_cruce.columns else None,
    )
else:
    fig_preview_cruce = px.scatter()

fig_preview_cruce.update_layout(
    template="plotly_dark",
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=False
)

layout = html.Div(
    children=[
        html.H2(
            "Panel General – Segmentación de Clientes y Patrones de Compra",
            className="page-title",
        ),
        html.P(
            "Explora las reglas de asociación (Apriori), los clústeres de clientes (K-Means) "
            "y el cruce entre ambos enfoques para descubrir patrones accionables.",
            className="page-subtitle",
        ),

        html.Div(
            className="card-grid-3",
            children=[
                dcc.Link(
                    href="/apriori",
                    className="glass-link",
                    children=html.Div(
                        className="glass-card",
                        children=[
                            html.Div("Apriori – Reglas de Asociación", className="card-title"),
                            html.Div(
                                "Soporte, confianza y elevación para encontrar productos que se compran juntos.",
                                className="card-subtitle",
                            ),
                            html.Div("Exploración global de reglas", className="card-pill"),
                            html.Div(
                                dcc.Graph(
                                    figure=fig_preview_apriori,
                                    config={"displayModeBar": False},
                                ),
                                className="graph-container",
                            ),
                        ],
                    ),
                ),
                dcc.Link(
                    href="/kmeans",
                    className="glass-link",
                    children=html.Div(
                        className="glass-card",
                        children=[
                            html.Div("K-Means – Segmentación de Clientes", className="card-title"),
                            html.Div(
                                "Distribución de clientes, comportamiento temporal y preferencias de compra.",
                                className="card-subtitle",
                            ),
                            html.Div("Exploración de clústeres", className="card-pill"),
                            html.Div(
                                dcc.Graph(
                                    figure=fig_preview_kmeans,
                                    config={"displayModeBar": False},
                                ),
                                className="graph-container",
                            ),
                        ],
                    ),
                ),
                dcc.Link(
                    href="/cruce",
                    className="glass-link",
                    children=html.Div(
                        className="glass-card",
                        children=[
                            html.Div("Cruce Apriori + K-Means", className="card-title"),
                            html.Div(
                                "Reglas de asociación segmentadas por clúster para diseñar acciones específicas.",
                                className="card-subtitle",
                            ),
                            html.Div("Patrones por segmento", className="card-pill"),
                            html.Div(
                                dcc.Graph(
                                    figure=fig_preview_cruce,
                                    config={"displayModeBar": False},
                                ),
                                className="graph-container",
                            ),
                        ],
                    ),
                ),
            ],
        ),
    ]
)
