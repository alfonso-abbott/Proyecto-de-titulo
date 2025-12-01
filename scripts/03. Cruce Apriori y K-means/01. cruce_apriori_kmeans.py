###################################################################################################################
###################################################################################################################
########################### 01. Cruce de Reglas Apriori con Segmentación por K-means ##############################
####################################### Asociación personalizada por perfil #######################################
###################################################################################################################
###################################################################################################################

###################################################################################################################
# OBJETIVO DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# Este script permite realizar un cruce entre los resultados del análisis de reglas de asociación (Apriori)
# y los clústeres generados por K-means, con el objetivo de obtener reglas personalizadas por perfil de cliente.
#
# Permite responder:
# - ¿Qué productos tienden a ser comprados en conjunto dentro de cada clúster?
# - ¿Cómo varían las reglas de asociación según los patrones de comportamiento de compra?
#
# Este cruce tiene utilidad para:
# - Construir sistemas de recomendación personalizados por segmento.
# - Generar insights de marketing más dirigidos y eficientes.
###################################################################################################################

###################################################################################################################
# SALIDA DEL SCRIPT
# -------------------------------------------------------------------------------------------------
# 📤 data/procesados/06. reglas_apriori_por_cluster.csv
###################################################################################################################

#################################### 📦 1. Librerías ####################################

import pandas as pd
import os

# 📁 Crear carpeta de salida
os.makedirs("./data/procesados", exist_ok=True)

#################################### 📥 2. Cargar datasets base ####################################

# 🛒 Relación orden-producto
df_op = pd.read_csv("./data/datos/order_products__prior.csv")

# 👥 Información de órdenes
df_orders = pd.read_csv("./data/datos/orders.csv")

# 📦 Productos (para nombres)
df_products = pd.read_csv("./data/datos/products.csv")

# 🧬 Clúster de cada usuario
df_clusters = pd.read_csv("./data/procesados/04. clientes_clusterizados.csv")

#################################### 🔗 3. Unión de datos ####################################

#################################### 🔁 3. Unión de datos con control de usuarios válidos ####################################

# 🧼 Filtrar solo órdenes donde user_id está presente
df_orders = df_orders.dropna(subset=["user_id"])

# 🎯 Filtrar solo usuarios que aparecen en los clústeres
usuarios_validos = df_clusters["user_id"].unique()
df_orders = df_orders[df_orders["user_id"].isin(usuarios_validos)]

# 🔗 Unir orden-producto con órdenes (para obtener user_id)
df_op = df_op.merge(df_orders[["order_id", "user_id"]], on="order_id", how="inner")

# 🔗 Agregar clúster de cada usuario
df_op = df_op.merge(df_clusters[["user_id", "cluster"]], on="user_id", how="inner")

# 🔗 Agregar nombre del producto
df_op = df_op.merge(df_products[["product_id", "product_name"]], on="product_id", how="left")

#################################### 🧹 4. Agrupación final de transacciones por clúster ####################################

# 🧼 Filtrar columnas necesarias
df_apriori_cluster = df_op[["order_id", "cluster", "product_name"]].dropna()

# 📊 Agrupar productos por orden y clúster
df_grouped = df_apriori_cluster.groupby(["cluster", "order_id"])["product_name"].apply(list).reset_index()

# ✅ Exportar
df_grouped.to_csv("./data/procesados/06. transacciones_apriori_por_cluster.csv", index=False)
print("✅ Archivo corregido y exportado: 06. transacciones_apriori_por_cluster.csv")

prueba = pd.read_csv("./data/procesados/06. transacciones_apriori_por_cluster.csv")
print(prueba["cluster"].value_counts())