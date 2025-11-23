##############################################################
# 🧺 Departamento más comprado por clúster
# Archivo: visualizacion_departamento_top.py
##############################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 Cargar dataset
df = pd.read_csv("./data/processed/clientes_clusterizados.csv")

# 🎯 Identificar columnas de departamentos (excluye las métricas generales)
departamentos = df.columns[6:-1]  # Asume que las últimas columnas antes de 'cluster' son los deptos

# 🔄 Transformar de ancho a largo
df_largo = df.melt(id_vars=["cluster"], value_vars=departamentos,
                   var_name="departamento", value_name="n_compras")

# 🎯 Seleccionar el departamento más comprado por clúster
df_top = df_largo.groupby(["cluster", "departamento"])["n_compras"].sum().reset_index()
df_top = df_top.sort_values(["cluster", "n_compras"], ascending=[True, False])
df_top_max = df_top.groupby("cluster").first().reset_index()

# 📊 Visualización
plt.figure(figsize=(10, 6))
sns.barplot(data=df_top_max, x="cluster", y="n_compras", hue="departamento", dodge=False)
plt.title("Departamento más comprado por Clúster", fontsize=14)
plt.xlabel("Clúster")
plt.ylabel("Cantidad total de compras")
plt.legend(title="Departamento", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("./output/clustering_departamento_top.png")
plt.show()
