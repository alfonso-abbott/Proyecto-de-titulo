###################################################################################################################
###################################################################################################################
############### 06. Preferencias de compra - Proporción de compras por departamento por clúster ###################
############################## Análisis relativo del comportamiento de compra por segmento ########################
###################################################################################################################
###################################################################################################################

###################################################################################################################
# OBJETIVO DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# Este script analiza la proporción (%) de compras realizadas en cada departamento para cada clúster.
#
# Permite responder:
# - ¿Qué tan diverso es el consumo por clúster?
# - ¿Qué departamentos son dominantes o específicos en cada grupo?
#
# Esta visualización es útil para:
# - Identificar patrones de preferencia de productos por perfil de cliente.
# - Definir estrategias de oferta diferenciada por segmento.
# - Detectar hábitos de compra relacionados con departamentos específicos.
###################################################################################################################

###################################################################################################################
# SALIDA DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# 📤 output/03. K-means/06. proporcion_departamento_por_cluster.png
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

#################################### 🧹 3. Seleccionar columnas de departamentos ####################################

# Se asume que las columnas de departamentos están entre la columna 6 y la penúltima (antes de 'cluster')
departamentos = df.columns[6:-1]

#################################### 🧮 4. Sumar compras por clúster y normalizar ####################################

# Agrupar por clúster y sumar cantidad de compras por departamento
df_sum = df.groupby("cluster")[departamentos].sum()

# Calcular proporción (porcentaje) dentro de cada clúster
df_prop = df_sum.div(df_sum.sum(axis=1), axis=0) * 100
df_prop = df_prop.round(2)

#################################### 🔁 5. Transformar datos para graficar ####################################

df_prop_reset = df_prop.reset_index().melt(
    id_vars="cluster",
    var_name="departamento",
    value_name="proporcion"
)

#################################### 📊 6. Visualización mejorada de barras por clúster ####################################

plt.figure(figsize=(14, 7))

# 🎨 Paleta personalizada por departamento
departamento_colores = {
    "Alcohol": "#8B0000",                # rojo oscuro
    "Bebes": "#FFB6C1",                  # rosado suave
    "Panaderia": "#D2B48C",              # marrón claro (pan)
    "Bebidas": "#1E90FF",                # azul fuerte
    "Desayuno": "#FFD700",              # dorado (cereal/miel)
    "Productos a granel": "#A0522D",     # marrón (granos)
    "Productos enlatados": "#708090",    # gris acero
    "Lacteos y huevos": "#FFFACD",       # amarillo pálido
    "Fiambres": "#DC143C",               # rojo intenso
    "Abarrotes y pastas": "#F4A460",     # arena
    "Congelados": "#4682B4",             # azul acero (frío)
    "Productos hogar": "#2F4F4F",        # gris pizarra oscuro
    "Productos internacionales": "#9ACD32", # verde lima
    "Carnes y mariscos": "#B22222",      # rojo carne
    "Faltantes": "#808080",              # gris
    "Otros": "#C0C0C0",                  # gris claro
    "Despensa": "#DAA520",               # dorado opaco
    "Cuidado personal": "#6495ED",       # azul claro
    "Mascotas": "#20B2AA",               # verde agua
    "Frutas y verduras": "#228B22",      # verde intenso
    "Snacks": "#FF69B4"                  # rosado vivo
}

# 📊 Gráfico de barras
ax = sns.barplot(data=df_prop_reset,
                 x="cluster",
                 y="proporcion",
                 hue="departamento",
                 palette=departamento_colores,
                 edgecolor="black",   # 🟢 Bordes marcados en las barras
                 linewidth=0.5)

# 🎯 Estilo general
plt.title("Proporción de compras por departamento por clúster", fontsize=16, weight='bold', loc='center', pad=15)
plt.xlabel("Clúster", fontsize=12)
plt.ylabel("Proporción (%)", fontsize=12)

# 🧱 Cuadrícula personalizada
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# 🧭 Ajustes de ejes
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
ax.set_ylim(0, df_prop_reset["proporcion"].max() + 5)  # márgen superior

# 🧾 Leyenda externa con mejor orden
plt.legend(title="Departamento",
           bbox_to_anchor=(1.02, 1),
           loc="upper left",
           ncol=2,
           frameon=True,
           borderpad=1,
           fontsize=9,
           title_fontsize=11)

# 🧩 Bordes de la figura
sns.despine(left=False, bottom=False)

# 🎯 Márgenes internos más cómodos
plt.tight_layout()


#################################### 💾 7. Guardar gráfico ####################################

plt.savefig("./output/03. K-means/06. Preferencias de compra. Proporción de compras por departamento por clúster.png", dpi=300)
plt.show()
