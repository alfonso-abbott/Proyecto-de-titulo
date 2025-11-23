#########################################################
# 📌 Dispersión PCA de los Clústeres de Clientes
# Archivo: visualizacion_dispersion_clustering.py
#########################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 📥 Cargar dataset ya clusterizado
df = pd.read_csv("./data/processed/clientes_clusterizados.csv")

# 🧼 Escalar datos (sin incluir user_id ni cluster)
X = df.drop(["user_id", "cluster"], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📉 Aplicar PCA para reducir a 2 dimensiones
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(X_scaled)

# 📌 Agregar resultado al dataframe
df["PCA1"] = pca_result[:, 0]
df["PCA2"] = pca_result[:, 1]

# 🎨 Graficar dispersión
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="cluster", palette="tab10", alpha=0.6, s=40)
plt.title("Dispersión de Clientes por Clúster (PCA)", fontsize=14)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Clúster")
plt.tight_layout()
plt.savefig("./output/clustering_dispersion_pca.png")
plt.show()
