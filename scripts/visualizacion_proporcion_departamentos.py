#################################################################
# 📊 Proporción de compras por departamento por clúster (stacked)
# Archivo: visualizacion_proporcion_departamentos.py
#################################################################

import pandas as pd
import matplotlib.pyplot as plt

# 📥 Cargar datos
df = pd.read_csv("./data/processed/clientes_clusterizados.csv")

# 🎯 Columnas de departamentos (asume estructura conocida)
departamentos = df.columns[6:-1]  # desde "alcohol" hasta antes de "cluster"

# 🔄 Agrupar por clúster y sumar compras por departamento
df_grouped = df.groupby("cluster")[departamentos].sum()

# 🔢 Calcular proporciones
df_prop = df_grouped.div(df_grouped.sum(axis=1), axis=0)

# 🎨 Visualización
df_prop.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab20")
plt.title("Proporción de compras por departamento por Clúster", fontsize=14)
plt.xlabel("Clúster")
plt.ylabel("Proporción de compras")
plt.legend(title="Departamento", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("./output/clustering_proporcion_departamentos.png")
plt.show()
