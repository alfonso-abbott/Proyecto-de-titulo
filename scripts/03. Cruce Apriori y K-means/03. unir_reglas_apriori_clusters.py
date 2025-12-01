###################################################################################################################
###################################################################################################################
######################## 03. Cruce Apriori y K-means - Consolidación de reglas por clúster ########################
######################################### Unión de archivos individuales de reglas ################################
###################################################################################################################
###################################################################################################################

###################################################################################################################
# OBJETIVO DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# Este script une todas las reglas Apriori generadas por clúster en un único archivo consolidado.
#
# Permite:
# - Comparar reglas entre clústeres.
# - Filtrar por lift, soporte o confianza de forma global o por segmento.
# - Visualizar insights personalizados por grupo de clientes.
###################################################################################################################

###################################################################################################################
# SALIDA DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# 📤 ./data/procesados/08. resumen_reglas_apriori_clusters.csv
###################################################################################################################

#################################### 📦 1. Librerías y configuración ####################################

import pandas as pd
import os
from glob import glob

# 📁 Crear carpeta de salida si no existe
os.makedirs("./data/procesados", exist_ok=True)

#################################### 📥 2. Cargar archivos de reglas por clúster ####################################

# 🧠 Todos los archivos CSV de reglas individuales
archivos = glob("./data/procesados/07. reglas_apriori_cluster_*.csv")

# 📊 Unir en un solo DataFrame
df_total = pd.DataFrame()

for archivo in archivos:
    cluster_id = int(os.path.basename(archivo).split("_")[-1].replace(".csv", ""))
    df = pd.read_csv(archivo)
    df["cluster"] = cluster_id
    df_total = pd.concat([df_total, df], ignore_index=True)

#################################### 💾 3. Guardar resultado unificado ####################################

df_total.to_csv("./data/procesados/08. resumen_reglas_apriori_clusters.csv", index=False)

print("✅ Reglas consolidadas guardadas en ./data/procesados/08. resumen_reglas_apriori_clusters.csv")
