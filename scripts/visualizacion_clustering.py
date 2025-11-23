##############################################################
# VISUALIZACIÓN DE CLÚSTERES CON PCA 🌈
# Archivo: visualizacion_clustering.py
##############################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 📥 Cargar dataset ya clusterizado
df_cluster = pd.read_csv("./data/processed/clientes_clusterizados.csv")

# 🎯 Eliminar columnas no numéricas para el PCA
X = df_cluster.drop(columns=["user_id", "cluster"])

# ⚖️ Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔍 Aplicar PCA a 2 dimensiones
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

# 🧷 Crear DataFrame con resultados de PCA y clúster
df_pca = pd.DataFrame()
df_pca["PCA_1"] = pca_result[:, 0]
df_pca["PCA_2"] = pca_result[:, 1]
df_pca["cluster"] = df_cluster["cluster"].astype(str)

# 🎨 Visualización
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x="PCA_1", y="PCA_2", hue="cluster", palette="Set2", s=30)
plt.title("Visualización de Clústeres de Clientes con PCA 🌐")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# 💾 Guardar gráfico
plt.savefig("./output/clusters_pca.png")
plt.show()
