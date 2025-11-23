#########################################################
# 📌 Distribución de usuarios por clúster
# Archivo: visualizacion_distribucion_cluster.py
#########################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 Cargar dataset
df = pd.read_csv("./data/processed/clientes_clusterizados.csv")

# 🎨 Configurar estilo
sns.set(style="whitegrid")

# 📊 Conteo de usuarios por clúster
cluster_counts = df["cluster"].value_counts().sort_index()

# 📈 Gráfico de barras
plt.figure(figsize=(8, 5))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="tab10")
plt.title("Distribución de Usuarios por Clúster", fontsize=14)
plt.xlabel("Clúster")
plt.ylabel("Cantidad de Usuarios")
plt.tight_layout()
plt.savefig("./output/clustering_distribucion_usuarios.png")
plt.show()
