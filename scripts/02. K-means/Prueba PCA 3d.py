import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

# 1. Cargar dataset clusterizado
df = pd.read_csv("./data/procesados/04. clientes_clusterizados.csv")

# 2. Separar variables (sin user_id ni cluster)
X = df.drop(["user_id", "cluster"], axis=1)

# 3. Escalar variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. PCA a 3 componentes
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_scaled)

df_pca_3d = pd.DataFrame(X_pca_3d, columns=["PCA1", "PCA2", "PCA3"])
df_pca_3d["cluster"] = df["cluster"]

# 5. Gráfico interactivo en 3D
fig = px.scatter_3d(
    df_pca_3d,
    x="PCA1",
    y="PCA2",
    z="PCA3",
    color="cluster",
    opacity=0.75,
    title="🧭 Visualización interactiva 3D de clusters con PCA",
)

fig.update_traces(marker=dict(size=4, line=dict(width=1, color="black")))

fig.update_layout(
    width=900,
    height=700,
    legend_title="Cluster"
)

# 6. Guardar HTML
fig.write_html("./output/03. K-means/Prueba PCA 3d.html")

print("Gráfico 3D generado correctamente.")
