############################################
# HEATMAP DE CONSUMO POR CLUSTER Y DEPARTAMENTO 🧯
# Archivo: visualizacion_heatmap_departamento.py
############################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 📥 Cargar archivo
df = pd.read_csv("./data/processed/clientes_clusterizados.csv")

# 🔎 Filtrar columnas de interés (solo departamentos + cluster)
departamentos = df.columns.difference(["user_id", "n_pedidos", "orden_max", "dia_promedio",
                                       "hora_promedio", "dias_entre_pedidos", "cluster"])

# Agrupar por cluster y promediar consumo por departamento
heatmap_data = df.groupby("cluster")[departamentos].mean()

# 🎨 Estilo visual
plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5)
plt.title("Consumo Promedio por Departamento y Cluster", fontsize=14)
plt.ylabel("Cluster")
plt.xlabel("Departamento")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 💾 Guardar imagen
plt.savefig("./output/heatmap_consumo_departamentos.png")
print("✅ Heatmap exportado: output/heatmap_consumo_departamentos.png")
