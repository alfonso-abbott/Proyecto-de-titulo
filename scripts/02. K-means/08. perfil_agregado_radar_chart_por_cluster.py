###################################################################################################################
###################################################################################################################
######################## 08. Perfil agregado - Radar Chart por clúster ############################################
################################### Visualización multivariable por grupo #########################################
###################################################################################################################
###################################################################################################################

###################################################################################################################
# OBJETIVO DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# Este script genera un gráfico radar que compara clústeres de clientes en múltiples dimensiones:
# - Total de pedidos
# - Hora promedio de compra
# - Total gastado aproximado (usaremos orden_max como proxy)
# - Preferencia por tipo de producto (departamentos)
#
# Permite observar perfiles agregados y responder:
# - ¿Qué clúster compra más?
# - ¿Cuál prefiere ciertos productos?
# - ¿En qué se diferencian los clústeres?
###################################################################################################################

###################################################################################################################
# SALIDA DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# 📤 output/03. K-means/06. perfil_radar_por_cluster.png
###################################################################################################################

#################################### 📦 1. Librerías y configuración ####################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 🎨 Estilo visual
sns.set(style="whitegrid", font_scale=1.1)

# 📁 Crear carpeta si no existe
os.makedirs("./output/03. K-means", exist_ok=True)

#################################### 📥 2. Cargar dataset clusterizado ####################################
df = pd.read_csv("./data/procesados/04. clientes_clusterizados.csv")

#################################### 🧮 3. Selección de variables para el radar chart ####################################
# 🔢 Columnas que se usarán en el gráfico radar
columnas_radar = [
    'n_pedidos',
    'hora_promedio',
    'orden_max',
    'Alcohol',
    'Panaderia',
    'Bebidas',
    'Lacteos y huevos',
    'Congelados',
    'Carnes y mariscos',
    'Frutas y verduras'
]

# 🧠 Cálculo del promedio por clúster
df_mean = df.groupby("cluster")[columnas_radar].mean()

# 🔄 Normalizar para radar chart (0 a 1)
df_norm = (df_mean - df_mean.min()) / (df_mean.max() - df_mean.min())

#################################### 📊 4. Construcción del gráfico radar (estilo mejorado) ####################################

# 🎯 Preparar estructura del radar
labels = columnas_radar
n_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
angles += angles[:1]  # cerrar el círculo

# 📐 Configurar figura con fondo sobrio
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
ax.set_facecolor("#f8f8f8")  # 🎨 fondo claro neutro

# 🎨 Paleta de colores diferenciada
colors = sns.color_palette("tab10", n_colors=len(df_norm))  # mayor contraste y diferenciación

# 🔁 Dibujar cada clúster
for idx, row in df_norm.iterrows():
    values = row.tolist()
    values += values[:1]  # cerrar forma
    ax.plot(angles, values, label=f'Clúster {idx}', linewidth=2, color=colors[idx])
    ax.fill(angles, values, color=colors[idx], alpha=0.15)  # sombra más sutil

# 🧭 Ajustes visuales elegantes
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11, color="#333333", fontweight='medium')
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="#555555", size=9)
plt.ylim(0, 1)

# ✏️ Título y leyenda
plt.title("Perfil de compra agregado por clúster (Radar Chart)", fontsize=16, weight='bold', pad=25)
plt.legend(loc='upper right',
           bbox_to_anchor=(1.25, 1.15),
           frameon=True,
           framealpha=0.9,
           edgecolor="#444444",
           fontsize=10,
           title="Clúster",
           title_fontsize=11)

# 🎯 Bordes más limpios
ax.spines["polar"].set_color("#444444")
ax.spines["polar"].set_linewidth(1)

# 💎 Disposición final
plt.tight_layout()


# 💾 Guardar
plt.savefig("./output/03. K-means/07. Perfil agregado. Radar Chart por clúster.png", dpi=300)
plt.show()
