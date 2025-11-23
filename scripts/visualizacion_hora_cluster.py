##############################################################
# 🕒 Promedio de hora de compra por clúster
# Archivo: visualizacion_hora_cluster.py
##############################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 Cargar dataset
df = pd.read_csv("./data/processed/clientes_clusterizados.csv")

# 🎨 Estilo
sns.set(style="whitegrid")

# 🔍 Agrupar por clúster y calcular media de la hora de compra
hora_promedio = df.groupby("cluster")["hora_promedio"].mean().reset_index()

# 📈 Gráfico
plt.figure(figsize=(8, 5))
sns.barplot(data=hora_promedio, x="cluster", y="hora_promedio", palette="coolwarm")
plt.title("Promedio de Hora de Compra por Clúster", fontsize=14)
plt.xlabel("Clúster")
plt.ylabel("Hora Promedio")
plt.tight_layout()
plt.savefig("./output/clustering_hora_promedio.png")
plt.show()
