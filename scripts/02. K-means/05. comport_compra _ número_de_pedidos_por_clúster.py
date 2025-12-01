###################################################################################################################
###################################################################################################################
########################### 07. Comportamiento de compra - Número de pedidos por clúster ##########################
################################### Análisis de intensidad de compras por segmento ################################
###################################################################################################################
###################################################################################################################

###################################################################################################################
# OBJETIVO DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# Este script analiza cómo varía la cantidad de pedidos realizados por los clientes en cada clúster.
#
# Permite responder:
# - ¿Hay clústeres con clientes más intensivos o frecuentes en sus compras?
# - ¿Cuál es la variabilidad dentro de cada clúster respecto al número de pedidos?
#
# Este análisis es clave para:
# - Identificar clientes frecuentes (potencial fidelización).
# - Reconocer perfiles ocasionales o de bajo volumen (posibles campañas de reactivación).
###################################################################################################################

###################################################################################################################
# SALIDA DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# 📤 output/03. K-means/07. distribucion_numero_pedidos_por_cluster.png
###################################################################################################################

########################################### 📦 1. Librerías y estilo ###############################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 🎨 Estilo visual moderno y profesional
sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1

# 📁 Asegurar carpeta de salida
os.makedirs("./output/03. K-means", exist_ok=True)

########################################### 📥 2. Cargar dataset clusterizado ######################################

df = pd.read_csv("./data/procesados/04. clientes_clusterizados.csv")

##################################### 📊 3. Visualización: Boxplot por clúster #####################################

plt.figure(figsize=(12, 6))

# 📦 Boxplot con líneas claras y paleta elegante
ax = sns.boxplot(data=df, 
                 x="cluster", 
                 y="n_pedidos", 
                 palette="pastel", 
                 linewidth=1.5,
                 fliersize=2)

# 📍 Agregar puntos individuales con jitter para mejor detalle
sns.stripplot(data=df, 
              x="cluster", 
              y="n_pedidos", 
              color='black', 
              alpha=0.3, 
              jitter=0.2, 
              size=2)

# 📝 Título y etiquetas
plt.title("Distribución del número de pedidos por clúster", fontsize=14, weight='bold', pad=15)
plt.xlabel("Clúster", fontsize=12)
plt.ylabel("Número de pedidos", fontsize=12)

# 🧭 Detalles visuales
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# 💾 Guardar gráfico
plt.savefig("./output/03. K-means/04. Comportamiento de compra. Número de pedidos por clúster.png", dpi=300)
plt.show()
