# pages/kmeans_page.py
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

import dash
from dash import html, dcc, Input, Output, callback

dash.register_page(
    __name__,
    path="/kmeans",
    name="K-Means",
    title="K-Means – Proyecto de Título",
)

RUTA = "./data/procesados/"
df = pd.read_csv(os.path.join(RUTA, "04. clientes_clusterizados.csv"))

# Aseguramos tipo entero de cluster
df["cluster"] = df["cluster"].astype(int)

# Columnas básicas
cols_base = [
    "user_id",
    "n_pedidos",
    "orden_max",
    "dia_promedio",
    "hora_promedio",
    "dias_entre_pedidos",
    "cluster",
]

# Columnas de departamentos (las que no son base)
dept_cols = [c for c in df.columns if c not in cols_base]

# PCA 2D sobre departamentos
X = df[dept_cols].values
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]


def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=10, t=40, b=40),
        legend_title_text="",
    )
    return fig


# Radar: agregados por cluster
agg = df.groupby("cluster").agg(
    {
        "n_pedidos": "mean",
        "orden_max": "mean",
        "hora_promedio": "mean",
        "dias_entre_pedidos": "mean",
        **{c: "mean" for c in dept_cols},
    }
)

# Normalizamos 0-1 para radar
agg_norm = (agg - agg.min()) / (agg.max() - agg.min())
radar_cols = [
    "n_pedidos",
    "hora_promedio",
    "orden_max",
] + dept_cols[:6]  # para que no sea infinito


layout = html.Div(
    children=[
        html.H2("K-Means – Segmentación de Clientes", className="page-title"),
        html.P(
            "Análisis de la distribución de clientes por clúster, comportamiento de compra "
            "y preferencias por departamento.",
            className="page-subtitle",
        ),

        # Filtro principal de cluster
        html.Div(
            className="glass-card",
            children=[
                html.Div("Filtros de segmentación", className="card-title"),
                html.Div(
                    className="filter-panel",
                    children=[
                        html.Div(
                            className="filter-field",
                            children=[
                                html.Div("Clúster seleccionado (para algunos gráficos)", className="filter-label"),
                                dcc.Dropdown(
                                    id="km-cluster",
                                    options=[
                                        {"label": f"Clúster {c}", "value": int(c)}
                                        for c in sorted(df["cluster"].unique())
                                    ],
                                    value=int(df["cluster"].mode()[0]),
                                    clearable=False,
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    "La mayoría de los gráficos muestran todos los clústeres; algunos se enfocan en el clúster seleccionado.",
                    className="helper-text",
                ),
            ],
        ),

        html.Br(),

        # Grid distribución + PCA
        html.Div(
            className="card-grid-2",
            children=[
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div(
                            "01. Distribución de clientes por clúster",
                            className="card-title",
                        ),
                        dcc.Graph(id="km-fig-dist", className="graph-full"),
                    ],
                ),
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div(
                            "02. Visualización de clústeres con PCA (2D)",
                            className="card-title",
                        ),
                        dcc.Graph(id="km-fig-pca", className="graph-full"),
                    ],
                ),
            ],
        ),

        html.Br(),

        # Grid comportamiento
        html.Div(
            className="card-grid-2",
            children=[
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div(
                            "03. Promedio de hora de compra por clúster",
                            className="card-title",
                        ),
                        dcc.Graph(id="km-fig-hora", className="graph-full"),
                    ],
                ),
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div(
                            "04. Distribución del número de pedidos por clúster",
                            className="card-title",
                        ),
                        dcc.Graph(id="km-fig-pedidos", className="graph-full"),
                    ],
                ),
            ],
        ),

        html.Br(),

        # Preferencias
        html.Div(
            className="card-grid-2",
            children=[
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div(
                            "05. Departamento más comprado por clúster",
                            className="card-title",
                        ),
                        dcc.Graph(id="km-fig-dept-top", className="graph-full"),
                    ],
                ),
                html.Div(
                    className="glass-card",
                    children=[
                        html.Div(
                            "06. Proporción de compras por departamento por clúster",
                            className="card-title",
                        ),
                        dcc.Graph(id="km-fig-dept-prop", className="graph-full"),
                    ],
                ),
            ],
        ),

        html.Br(),

        html.Div(
            className="glass-card",
            children=[
                html.Div(
                    "07. Perfil agregado por clúster (Radar Chart normalizado)",
                    className="card-title",
                ),
                dcc.Graph(id="km-fig-radar", className="graph-full"),
            ],
        ),
    ]
)


@callback(
    Output("km-fig-dist", "figure"),
    Input("km-cluster", "value"),
)
def fig_distribucion(_cluster):
    conteo = df["cluster"].value_counts().sort_index()
    fig = px.bar(
        x=conteo.index.astype(str),
        y=conteo.values,
        labels={"x": "Clúster", "y": "Cantidad de clientes"},
        text=conteo.values,
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(title="Distribución de clientes por clúster (k=5)")
    return style_fig(fig)


@callback(
    Output("km-fig-pca", "figure"),
    Input("km-cluster", "value"),
)
def fig_pca(cluster_sel):
    fig = px.scatter(
        df.sample(min(20000, len(df))),
        x="pca1",
        y="pca2",
        color="cluster",
        labels={
            "pca1": "Componente principal 1",
            "pca2": "Componente principal 2",
            "cluster": "Clúster",
        },
        opacity=0.6,
    )
    fig.update_layout(title="Visualización de clústeres de clientes con PCA (2D)")
    return style_fig(fig)


@callback(
    Output("km-fig-hora", "figure"),
    Input("km-cluster", "value"),
)
def fig_hora(_cluster):
    hora = df.groupby("cluster")["hora_promedio"].mean().round(2)
    fig = px.bar(
        x=hora.index.astype(str),
        y=hora.values,
        text=hora.values,
        labels={"x": "Clúster", "y": "Hora promedio de compra"},
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(title="Promedio de hora de compra por clúster")
    return style_fig(fig)


@callback(
    Output("km-fig-pedidos", "figure"),
    Input("km-cluster", "value"),
)
def fig_pedidos(_cluster):
    fig = px.box(
        df,
        x="cluster",
        y="n_pedidos",
        points="all",
        labels={"cluster": "Clúster", "n_pedidos": "Número de pedidos"},
    )
    fig.update_layout(title="Distribución del número de pedidos por clúster")
    return style_fig(fig)


@callback(
    Output("km-fig-dept-top", "figure"),
    Input("km-cluster", "value"),
)
def fig_dept_top(_cluster):
    # sumamos compras por cluster y departamento
    suma = df.groupby("cluster")[dept_cols].sum()
    # identificamos el dept más comprado por cluster (pero graficamos todo "Frutas y verduras")
    if "Frutas y verduras" in dept_cols:
        valores = suma["Frutas y verduras"]
    else:
        # en caso de no existir, usamos el dept con mayor suma global
        d_global = suma.sum().idxmax()
        valores = suma[d_global]
    fig = px.bar(
        x=valores.index.astype(str),
        y=valores.values,
        text=valores.values.astype(int),
        labels={"x": "Clúster", "y": "Cantidad total de compras"},
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(title="Departamento más comprado por clúster")
    return style_fig(fig)


@callback(
    Output("km-fig-dept-prop", "figure"),
    Input("km-cluster", "value"),
)
def fig_dept_prop(_cluster):
    suma = df.groupby("cluster")[dept_cols].sum()
    propor = (suma.T / suma.T.sum()).T * 100
    propor = propor.reset_index().melt(id_vars="cluster", var_name="Departamento", value_name="Proporción")
    fig = px.bar(
        propor,
        x="cluster",
        y="Proporción",
        color="Departamento",
        barmode="group",
        labels={"cluster": "Clúster", "Proporción": "Proporción (%)"},
    )
    fig.update_layout(title="Proporción de compras por departamento por clúster")
    return style_fig(fig)


@callback(
    Output("km-fig-radar", "figure"),
    Input("km-cluster", "value"),
)
def fig_radar(_cluster):
    categorias = radar_cols
    fig = go.Figure()
    for cluster, fila in agg_norm.iterrows():
        valores = fila[categorias].tolist()
        fig.add_trace(
            go.Scatterpolar(
                r=valores + [valores[0]],
                theta=categorias + [categorias[0]],
                name=f"Clúster {cluster}",
                fill="toself",
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Perfil de compra agregado por clúster (valores normalizados)",
    )
    return style_fig(fig)
