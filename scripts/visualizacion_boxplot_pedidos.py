#############################################
# 📦 BOXPLOT: Distribución de pedidos por clúster
# Archivo: visualizacion_boxplot_pedidos.py
#############################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🎯 Cargar dataset con clusters
df = pd.read_csv("./data/processed/clientes_clusterizados.csv")

# 🎨 Estilo visual
sns.set(style="whitegrid")

# 📊 Gráfico de cajas por clúster
plt.figure(figsize=(10, 6))
sns.boxplot(x="cluster", y="n_pedidos", data=df, palette="Set2")

# 🏷️ Personalización
plt.title("📦 Distribución del número de pedidos por clúster")
plt.xlabel("Clúster")
plt.ylabel("Número de pedidos")
plt.tight_layout()

# 💾 Guardar (opcional)
# plt.savefig("./output/boxplot_pedidos_cluster.png")

# 👁️ Mostrar
plt.show()
