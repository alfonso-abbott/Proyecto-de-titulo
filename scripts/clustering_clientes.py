
################################################################
##################  ANÁLISIS DE CLUSTERING DE CLIENTES 🤖
##################  Archivo: clustering_clientes.py
################################################################

#################################### 📦 1- Librerías de datos ####################################

import pandas as pd
import numpy as np

# 📏 Procedimiento de escalado

from sklearn.preprocessing import StandardScaler

# 🔗 Clustering
from sklearn.cluster import KMeans

# 🧠 Evaluación
from sklearn.metrics import silhouette_score

# 📊 Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# ⚠️ Warnings
import warnings
warnings.filterwarnings("ignore")


#################################### 📥 2- Cargar dataset ####################################


dtypes = {
    "user_id": "int32",
    "order_id": "int32",
    "order_number": "int8",
    "order_dow": "int8",
    "order_hour_of_day": "int8",
    "days_since_prior_order": "float32",
    "department": "category"
}

## Se definen las categorías para no saturar y optimizar la memoria RAM 

## Cargar CSV optimizando memoria
df_clientes = pd.read_csv("./data/processed/clientes_clustering.csv", dtype=dtypes)



# 👁️ Primeras filas
print("📋 Primeras filas:")
print(df_clientes.head())

# 🔢 Dimensiones
print("\n🔢 Dimensiones del dataset:", df_clientes.shape)

# ℹ️ Información general
print("\nℹ️ Tipos de datos:")
print(df_clientes.info())


#################################### 🔄 3- Agrupar por cliente ####################################


# 📊 Estadísticas base por usuario
df_agrupado = df_clientes.groupby("user_id").agg({
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

# 🏷️ Pivot de departamentos comprados por usuario
pivot_dept = pd.crosstab(df_clientes["user_id"], df_clientes["department"]).reset_index()

# 🔗 Unir ambas estructuras
df_final = df_agrupado.merge(pivot_dept, on="user_id")

# 👁️ Verificación
print("\n✅ Dataset preparado para clustering:")
print(df_final.head())
print("\n🔢 Dimensiones finales:", df_final.shape)



###################### 🔄 4- Normalización y aplicación de K-Means ######################


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 🧼 Eliminar columna user_id para escalar
X = df_final.drop("user_id", axis=1)

# ⚖️ Escalar variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📍 Aplicar K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
df_final["cluster"] = kmeans.fit_predict(X_scaled)

# 👁️ Visualizar distribución de clientes por grupo
print("\n📊 Distribución por cluster:")
print(df_final["cluster"].value_counts())

# 💾 Exportar resultados
df_final.to_csv("./data/processed/clientes_clusterizados.csv", index=False)
print("\n✅ Archivo exportado: clientes_clusterizados.csv")
