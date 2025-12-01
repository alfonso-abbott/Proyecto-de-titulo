###################################################################################################################
###################################################################################################################
#################### 04. Comportamiento de compra - Promedio hora de compra por clúster ###########################
###################################### Análisis de patrones temporales de compra ##################################
###################################################################################################################
###################################################################################################################

###################################################################################################################
# OBJETIVO DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# Este script analiza el promedio de hora en que los clientes de cada clúster realizan sus compras.
#
# Permite responder:
# - ¿Hay clústeres con preferencias horarias específicas?
# - ¿Existen perfiles de clientes que compran más de noche, de tarde o en la mañana?
#
# Este patrón temporal puede ser útil para:
# - Diseñar estrategias de marketing por hora del día.
# - Preparar campañas de notificación o delivery con mejor respuesta.
###################################################################################################################

###################################################################################################################
# SALIDA DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# 📤 output/03. K-means/03. promedio_hora_por_cluster.png
###################################################################################################################

#################################### 📦 1. Librerías y configuración ####################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 🎨 Estilo visual moderno
sns.set(style="whitegrid", font_scale=1.2)

# 📁 Crear carpeta si no existe
os.makedirs("./output/03. K-means", exist_ok=True)

#################################### 📥 2. Cargar dataset clusterizado ####################################

df = pd.read_csv("./data/procesados/04. clientes_clusterizados.csv")

#################################### 📊 3. Calcular promedio de hora por clúster ####################################

promedios_hora = df.groupby("cluster")["hora_promedio"].mean().round(2).sort_index()

#################################### 🎨 4. Gráfico de barras con valores visibles ####################################

plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    x=promedios_hora.index,
    y=promedios_hora.values,
    palette="Set2",
    edgecolor="black"
)

# Mostrar los valores sobre cada barra
for index, value in enumerate(promedios_hora.values):
    plt.text(index, value + 0.2, str(value), ha='center', va='bottom', fontsize=11, weight='bold')

plt.title("Promedio de hora de compra por clúster", fontsize=15, weight="bold")
plt.xlabel("Cluster", fontsize=12)
plt.ylabel("Hora promedio", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(range(0, 25, 1))  # Rango de 0 a 24 para claridad
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# 📸 Guardar gráfico
plt.savefig("./output/03. K-means/03. Comportamiento de compra. Promedio de hora de compra por clúster.png", dpi=300)
plt.close()
