########################################
# ANÁLISIS DE REGLAS DE ASOCIACIÓN 🛍️
# Archivo: analisis_apriori.py
########################################

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os

# 📥 Cargar dataset procesado
df_apriori = pd.read_csv("./data/processed/transacciones_apriori.csv")

# ✂️ Filtrar productos más frecuentes (top 100)
productos_frecuentes = df_apriori["product_name"].value_counts().head(100).index.tolist()
df_filtrado = df_apriori[df_apriori["product_name"].isin(productos_frecuentes)]

# ✂️ Limitar número de órdenes (20.000)
ordenes_muestras = df_filtrado["order_id"].unique()[:20000]
df_filtrado = df_filtrado[df_filtrado["order_id"].isin(ordenes_muestras)]

# 🔁 Agrupar productos por orden
transacciones = df_filtrado.groupby("order_id")["product_name"].apply(list).tolist()

# 🧰 Codificar transacciones
te = TransactionEncoder()
te_ary = te.fit(transacciones).transform(transacciones)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 📊 Generar conjuntos frecuentes con Apriori
frecuentes = apriori(df_encoded, min_support=0.005, use_colnames=True)

# 📌 Generar reglas de asociación
reglas = association_rules(frecuentes, metric="lift", min_threshold=1.0)

# 🔍 Mostrar reglas principales
print("\n📋 Reglas generadas:")
print(reglas[["antecedents", "consequents", "support", "confidence", "lift"]].head())

# 💾 Guardar resultados
os.makedirs("./output", exist_ok=True)
reglas.to_csv("./output/reglas_apriori.csv", index=False)
print("\n✅ Reglas de asociación exportadas a: output/reglas_apriori.csv")
