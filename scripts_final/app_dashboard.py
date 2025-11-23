# 📊 DASHBOARD INTERACTIVO DE SEGMENTACIÓN Y REGLAS DE ASOCIACIÓN

import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# 🔹 Inicializar app
app = dash.Dash(__name__)
app.title = "Dashboard de Segmentación y Reglas"

# 📥 Cargar datos
df_clusters = pd.read_csv("./data/processed/clientes_clusterizados.csv")
df_reglas = pd.read_csv("./output/reglas_apriori.csv")

# 🎯 COLORES
colores_cluster = px.colors.qualitative.Set2
colores_barra = px.colors.qualitative.Pastel
colores_red = "lightblue"

# 1️⃣ Gráfico: Clientes por clúster
fig_cluster_count = px.histogram(df_clusters, x="cluster", color="cluster",
    title="Distribución de Clientes por Clúster",
    labels={"cluster": "Clúster", "count": "Cantidad"},
    color_discrete_sequence=colores_cluster
)

# 2️⃣ Gráfico: Hora promedio por clúster
hora_prom = df_clusters.groupby("cluster")["hora_promedio"].mean().reset_index()
fig_hora_prom = px.bar(hora_prom, x="cluster", y="hora_promedio",
    title="Hora Promedio de Compra por Clúster",
    labels={"hora_promedio": "Hora Promedio", "cluster": "Clúster"},
    color="cluster", color_discrete_sequence=colores_barra
)

# 3️⃣ Gráfico: Heatmap consumo por departamento y clúster
departamentos = df_clusters.drop(columns=["user_id", "n_pedidos", "orden_max", "dia_promedio", "hora_promedio", "dias_entre_pedidos", "cluster"])
departamentos["cluster"] = df_clusters["cluster"]
heat_data = departamentos.groupby("cluster").mean().round(2)

fig_heatmap = go.Figure(data=go.Heatmap(
    z=heat_data.values,
    x=heat_data.columns,
    y=heat_data.index,
    colorscale="YlGnBu"
))
fig_heatmap.update_layout(
    title="Consumo Promedio por Departamento y Clúster",
    xaxis_title="Departamento",
    yaxis_title="Clúster"
)

# 4️⃣ Gráfico: Soporte vs confianza
fig_scatter = px.scatter(df_reglas, x="support", y="confidence", size="lift", color="lift",
    title="Reglas de Asociación: Soporte vs Confianza",
    labels={"support": "Soporte", "confidence": "Confianza", "lift": "Lift"},
    color_continuous_scale="Viridis"
)

# 5️⃣ Red de reglas de asociación
G = nx.DiGraph()
for _, row in df_reglas.iterrows():
    antecedents = row["antecedents"].replace("frozenset({'", "").replace("'})", "").split("', '")
    consequents = row["consequents"].replace("frozenset({'", "").replace("'})", "").split("', '")
    for ant in antecedents:
        for con in consequents:
            G.add_edge(ant, con, weight=row["lift"])

pos = nx.spring_layout(G, seed=42)

edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"),
    hoverinfo="none", mode="lines")

node_x, node_y, node_text = [], [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

node_trace = go.Scatter(
    x=node_x, y=node_y, mode="markers+text",
    text=node_text, textposition="top center",
    hoverinfo="text",
    marker=dict(size=10, color=colores_red, line_width=1)
)

fig_red = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
    title=dict(text="Red Interactiva de Reglas de Asociación", font=dict(size=18)),
    showlegend=False, hovermode="closest",
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
))

# 🧱 Layout final
app.layout = html.Div(style={"fontFamily": "Arial", "padding": "10px"}, children=[
    html.H1("📊 Dashboard Interactivo: Segmentación y Reglas de Asociación", style={"textAlign": "center"}),

    html.Div([html.H2("1️⃣ Distribución de Clientes por Clúster"), dcc.Graph(figure=fig_cluster_count)]),
    html.Div([html.H2("2️⃣ Hora Promedio de Compra por Clúster"), dcc.Graph(figure=fig_hora_prom)]),
    html.Div([html.H2("3️⃣ Consumo Promedio por Departamento y Clúster"), dcc.Graph(figure=fig_heatmap)]),
    html.Div([html.H2("4️⃣ Reglas de Asociación: Soporte vs Confianza"), dcc.Graph(figure=fig_scatter)]),
    html.Div([html.H2("5️⃣ Red de Reglas de Asociación (Interactiva)"), dcc.Graph(figure=fig_red)])
])

# 🚀 Ejecutar
if __name__ == '__main__':
    app.run(debug=True)
