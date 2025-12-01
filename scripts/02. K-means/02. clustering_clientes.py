###################################################################################################################
###################################################################################################################
######################################### 2- Análisis final de clustering de clientes #############################
###################################################################################################################
###################################################################################################################

###################################################################################################################
# OBJETIVO GENERAL DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# Este script aplica el algoritmo K-Means con k = 5 (validado previamente) para segmentar clientes 
# de un supermercado online en grupos de comportamiento homogéneo. 
#
# El análisis se basa en un dataset procesado a nivel de cliente, que incluye:
# - Variables de comportamiento temporal (frecuencia, recencia, horarios).
# - Frecuencia de compra por departamento (bakery, beverages, dairy, etc.).
#
# Este script genera los archivos base para visualización y dashboard, guardando:
#  - CSV con el dataset clusterizado (detallado por cliente).
#  - CSV reducido con solo user_id y cluster.
#  - Gráfico de barras con la distribución de clientes por cluster.
#
# SALIDAS DEL SCRIPT:
# -------------------------------------------------------------------------------------------------
# 1) ./data/procesados/04. clientes_clusterizados.csv
# 2) ./data/procesados/04T. clientes_clusterizados_reducido.csv
# 3) ./output/03. K-means/02.1 distribucion_clusters.png
###################################################################################################################


#################################### 📦 1- Librerías ####################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


#################################### 📁 2- Preparar carpetas ####################################

os.makedirs("./data/procesados", exist_ok=True)
os.makedirs("./output/03. K-means", exist_ok=True)


#################################### 📥 3- Cargar dataset de clientes ####################################

df = pd.read_csv("./data/procesados/02. clientes_clustering.csv")


#################################### 🔄 4- Construcción del dataset a nivel de cliente ####################################

# 4.1 Métricas agregadas por cliente
df_agrupado = df.groupby("user_id").agg({
    "order_id": "nunique",
    "order_number": "max",
    "order_dow": "mean",
    "order_hour_of_day": "mean",
    "days_since_prior_order": "mean"
}).rename(columns={
    "order_id": "n_pedidos",
    "order_number": "orden_max",
    "order_dow": "dia_promedio",
    "order_hour_of_day": "hora_promedio",
    "days_since_prior_order": "dias_entre_pedidos"
}).reset_index()

# 4.2 Pivot de frecuencia de compra por departamento
pivot_dept = pd.crosstab(df["user_id"], df["department"]).reset_index()

# 4.3 Unión de ambos datasets
df_final = df_agrupado.merge(pivot_dept, on="user_id")


#################################### ⚖️ 5- Escalado de variables ####################################

X = df_final.drop("user_id", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#################################### 📌 6- Aplicar K-Means definitivo ####################################

# k validado = 5
kmeans = KMeans(n_clusters=5, random_state=42)
df_final["cluster"] = kmeans.fit_predict(X_scaled)


#################################### 📊 7- Visualización de distribución de clusters ####################################

# Conteo por cluster
counts = df_final["cluster"].value_counts().sort_index()

# Estilo visual general
sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))

# Paleta más estética
palette = sns.color_palette("husl", len(counts))

# Crear gráfico
bars = sns.barplot(
    x=counts.index,
    y=counts.values,
    palette=palette,
    edgecolor="black",
    linewidth=1.2
)

# Etiquetas de valores sobre cada barra
for i, v in enumerate(counts.values):
    plt.text(
        i,
        v + max(counts.values) * 0.01,
        f"{v:,}",          # separador de miles
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

# Títulos y texto
plt.title("Distribución de clientes por cluster (k=5)", fontsize=18, weight="bold")
plt.xlabel("Cluster", fontsize=14)
plt.ylabel("Cantidad de clientes", fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Limpieza visual
sns.despine()

plt.tight_layout()
plt.savefig("./output/03. K-means/01. distribucion_clusters.png", dpi=300)
plt.close()


#################################### 💾 8- Exportar resultados ####################################

# 8.1 Dataset completo (con variables y cluster asignado)
df_final.to_csv("./data/procesados/04. clientes_clusterizados.csv", index=False)

# 8.2 Dataset reducido (solo user_id y cluster)
df_final[["user_id", "cluster"]].to_csv("./data/procesados/05. clientes_clusterizados_reducido.csv", index=False)

print("✅ Script ejecutado con éxito. Clusters generados y archivos exportados.")