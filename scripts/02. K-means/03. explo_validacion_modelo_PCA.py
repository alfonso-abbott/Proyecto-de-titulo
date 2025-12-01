###################################################################################################################
###################################################################################################################
######################## 03. Exploración y validación del modelo - Visualización con PCA ##########################
###################################### Reducción de dimensionalidad para clustering ################################
###################################################################################################################
###################################################################################################################

###################################################################################################################
# OBJETIVO GENERAL DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# Este script genera una visualización exploratoria de los clientes en el espacio reducido por
# PCA (Análisis de Componentes Principales), coloreados según su clúster asignado por K-Means.
#
# La visualización ayuda a:
# - Ver si los grupos definidos por K-Means tienen una separación razonable.
# - Explorar posibles traslapes o formas de agrupación.
# - Complementar las métricas de validación ya calculadas (inercia, silueta, etc.).
#
# Este gráfico es útil tanto para el informe como para el dashboard final del proyecto.
###################################################################################################################

###################################################################################################################
# SALIDA PRINCIPAL
# -------------------------------------------------------------------------------------------------
# 📍 output/03. K-means/02. clustering_PCA.png
###################################################################################################################

#################################### 📦 1. Librerías y configuración ####################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 📉 PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 🎨 Estilo visual moderno
import matplotlib.ticker as mticker
sns.set(style="whitegrid", font_scale=1.2)

# 📁 Crear carpeta si no existe
os.makedirs("./output/03. K-means", exist_ok=True)

#################################### 📥 2. Cargar dataset clusterizado ####################################

# Dataset completo con todas las variables originales y el clúster asignado
df = pd.read_csv("./data/procesados/04. clientes_clusterizados.csv")

# Separar variables numéricas
X = df.drop(["user_id", "cluster"], axis=1)

# Escalar las variables para PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#################################### 📉 3. Aplicar PCA para reducción a 2 componentes ####################################

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Combinar con los clústeres para graficar
df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
df_pca["cluster"] = df["cluster"]

#################################### 🎨 4. Visualización: Clustering con PCA ####################################

plt.figure(figsize=(10, 7))
palette = sns.color_palette("husl", df_pca["cluster"].nunique())

scatter = sns.scatterplot(
    x="PCA1", y="PCA2", hue="cluster", data=df_pca,
    palette=palette, alpha=0.6, edgecolor="black", s=60
)

plt.title("Visualización de clusters de clientes con PCA (2D)", fontsize=15, weight="bold")
plt.xlabel("Componente Principal 1", fontsize=12)
plt.ylabel("Componente Principal 2", fontsize=12)
plt.legend(title="Cluster", loc="best", frameon=True)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()

# 📸 Guardar gráfico
plt.savefig("./output/03. K-means/01. Exporación del modelo . Distribución de clientes por clúster.png", dpi=300)
plt.close()
