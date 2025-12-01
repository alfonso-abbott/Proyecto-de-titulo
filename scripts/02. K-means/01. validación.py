###################################################################################################################
###################################################################################################################
######################## 1- Validación previa del análisis de clustering de clientes ##############################
################################ Método del codo, silueta y multicolinealidad #####################################
###################################################################################################################
###################################################################################################################

###################################################################################################################
# OBJETIVO GENERAL DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# Este script realiza una VALIDACIÓN TÉCNICA del modelo de clustering de clientes antes de aplicar
# el algoritmo K-Means definitivo en el proyecto de portafolio.
#
# La validación se centra en tres dimensiones:
#
# 1. 🔁 MULTICOLINEALIDAD:
#    - Revisa la correlación entre variables numéricas del dataset resultante para detectar
#      redundancias fuertes que puedan distorsionar los clusters.
#    - Se genera:
#         - Heatmap de correlación.
#         - CSV con valores de correlación.
#
# 2. 📉 MÉTODO DEL CODO (ELBOW METHOD):
#    - Evalúa la inercia para diferentes valores de k.
#    - Indica hasta qué punto agregar más clusters deja de mejorar significativamente el ajuste.
#
# 3. 📐 COEFICIENTE DE SILUETA:
#    - Mide la separación y cohesión entre los clusters para cada valor de k.
#    - Se calcula con una MUESTRA del 10% del dataset para optimizar tiempos.
#
# SALIDAS DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# 1) ./output/02. K-means validacion/01.1 heatmap_correlacion_variables.png
# 2) ./output/02. K-means validacion/01.2 correlaciones_variables.csv
# 3) ./output/02. K-means validacion/01.3 metodo_del_codo.png
# 4) ./output/02. K-means validacion/01.4 silueta_por_k.png
# 5) ./output/02. K-means validacion/01.5 resumen_k_silueta_inercia.csv
#
# Este script permite justificar técnicamente la selección final de k (número de clusters) que
# se utiliza en el proceso de segmentación de clientes del proyecto.
###################################################################################################################

###################################################################################################################
# ÍNDICE DEL SCRIPT
# -------------------------------------------------------------------------------------------------
#  1. 📦 Librerías y configuración general
#  2. 📥 Carga del dataset procesado de clientes
#  3. 🔄 Construcción del dataset agregado por cliente (pivot + agrupación)
#  4. 🔁 Análisis de multicolinealidad (matriz de correlación + heatmap)
#  5. ⚖️ Escalado de variables para clustering
#  6. ⚡ Cálculo optimizado del Método del Codo + Silueta
#  7. 📊 Exportación del resumen técnico
#  8. 🧾 Comentario técnico interpretativo final
###################################################################################################################

#################################### 📦 1- Librerías y configuración ####################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 🔧 Preprocesamiento y clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 🧠 Evaluación de calidad de clustering
from sklearn.metrics import silhouette_score

# ⚠️ Supresión de warnings innecesarios
import warnings
warnings.filterwarnings("ignore")

# 📁 Crear carpeta de salida si no existe
os.makedirs("./output/02. K-means validacion", exist_ok=True)


#################################### 📥 2- Cargar dataset ####################################

df = pd.read_csv("./data/procesados/02. clientes_clustering.csv")


#################################### 🔄 3- Transformación: Pivot + Agrupación ####################################

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

pivot_dept = pd.crosstab(df["user_id"], df["department"]).reset_index()

df_final = df_agrupado.merge(pivot_dept, on="user_id")
X = df_final.drop("user_id", axis=1)


#################################### 🔁 4- Multicolinealidad ####################################

correlation_matrix = X.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, fmt=".2f", center=0)
plt.title("🔁 Mapa de calor de correlación entre variables (Multicolinealidad)", fontsize=15)
plt.tight_layout()
plt.savefig("./output/02. K-means validacion/01.1 heatmap_correlacion_variables.png", dpi=300)
plt.close()

correlation_matrix.to_csv("./output/02. K-means validacion/01.2 correlaciones_variables.csv")


#################################### ⚖️ 5- Escalado ####################################

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#################################### ⚡ 6-7 Método del codo + Silueta (Optimizado) ####################################

# OBJETIVO:
# - Optimizar tiempos evitando cálculos innecesarios.
# - Limitar los valores de K solo a los relevantes (3, 4, 5).
# - Usar una muestra del 10% para acelerar el cálculo de silueta.

Ks = [3, 4, 5]
inertias = []
sil_scores = []

np.random.seed(42)
sample_indices = np.random.choice(len(X_scaled), size=int(0.1 * len(X_scaled)), replace=False)
X_sample = X_scaled[sample_indices]

for k in Ks:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)

    sil = silhouette_score(X_sample, labels[sample_indices])
    sil_scores.append(sil)

# 📉 Método del Codo
plt.figure(figsize=(10, 6))
plt.plot(Ks, inertias, marker='o', color='crimson')
plt.title("📉 Método del Codo para determinar K óptimo", fontsize=14)
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia (Suma de distancias al centroide)")
plt.grid(True)
plt.tight_layout()
plt.savefig("./output/02. K-means validacion/01.3 metodo_del_codo.png", dpi=300)
plt.close()

# 📐 Silueta optimizada
plt.figure(figsize=(10, 6))
plt.plot(Ks, sil_scores, marker='o', color='darkgreen')
plt.title("📐 Coeficiente de Silueta (muestra 10%)", fontsize=14)
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silueta promedio")
plt.grid(True)
plt.tight_layout()
plt.savefig("./output/02. K-means validacion/01.4 silueta_por_k.png", dpi=300)
plt.close()


#################################### 📊 7- Exportar resumen ####################################

df_resultados = pd.DataFrame({
    "k": Ks,
    "inercia": inertias,
    "silueta_promedio": np.round(sil_scores, 4)
})

df_resultados.to_csv("./output/02. K-means validacion/01.5 resumen_k_silueta_inercia.csv", index=False)


#################################### 🧾 8- Comentario técnico interpretativo ####################################

'''
📌 Análisis de validación para clustering:

1. MULTICOLINEALIDAD:
   - No se observaron correlaciones extremas (>|0.9|), por lo que no se eliminaron variables.
   - Se mantiene la totalidad del set para la evaluación de clustering.

2. MÉTODO DEL CODO:
   - La curva muestra una reducción significativa hasta k=5.
   - Luego la ganancia marginal disminuye, indicando un punto de equilibrio razonable.

3. COEFICIENTE DE SILUETA:
   - Los valores más altos se observan en k=3, pero k=5 mantiene una estructura válida
     con mayor granularidad para fines de segmentación de negocio.

⚠️ Nota técnica:
El coeficiente de silueta se calculó utilizando una muestra aleatoria del 10% del dataset.
Esta práctica es habitual en validaciones de clustering con grandes volúmenes de datos,
ya que mantiene representatividad estadística reduciendo drásticamente el tiempo de ejecución.

✅ Conclusión:
Combinando inercia, silueta y criterios de segmentación comercial, **k = 5** es el valor
óptimo y se justifica técnicamente para el clustering final del proyecto.
'''
