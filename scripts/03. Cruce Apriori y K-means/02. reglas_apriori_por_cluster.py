###################################################################################################################
############### 07. Cruce Apriori y K-means - Reglas de asociación por clúster (con muestreo aleatorio) ##########
###################################################################################################################

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import os
import ast

# 📁 Crear carpeta si no existe
os.makedirs("./data/procesados", exist_ok=True)

# 📥 Cargar dataset
df = pd.read_csv("./data/procesados/06. transacciones_apriori_por_cluster.csv")

# 🎯 Tamaño de muestra por clúster
N_MUESTRA = 10000  # Puedes ajustar este valor

# 🔁 Iterar por cada clúster
for cluster_id in sorted(df["cluster"].unique()):
    print(f"\n🔍 Procesando clúster {cluster_id}...")

    # 📄 Filtrar las transacciones del clúster
    df_cluster = df[df["cluster"] == cluster_id].copy()

    # 🧪 Aplicar muestreo aleatorio
    df_muestra = df_cluster.sample(n=min(N_MUESTRA, len(df_cluster)), random_state=42)

    # 🧼 Convertir strings a listas
    transacciones = df_muestra["product_name"].apply(ast.literal_eval).tolist()

    if len(transacciones) == 0:
        print(f"⚠️ Sin transacciones para clúster {cluster_id}, se omite.")
        continue

    # 🧠 Binarizar con TransactionEncoder
    try:
        te = TransactionEncoder()
        df_bin = pd.DataFrame(te.fit(transacciones).transform(transacciones), columns=te.columns_)
    except MemoryError as e:
        print(f"❌ Error de memoria en clúster {cluster_id}: {e}")
        continue

    # 📊 Apriori
    itemsets = apriori(df_bin, min_support=0.005, use_colnames=True)

    if itemsets.empty:
        print(f"⚠️ Sin itemsets frecuentes en clúster {cluster_id}.")
        continue

    reglas = association_rules(itemsets, metric="confidence", min_threshold=0.15)

    if reglas.empty:
        print(f"⚠️ Sin reglas generadas para clúster {cluster_id}.")
        continue

    reglas["cluster"] = cluster_id

    # 💾 Guardar reglas por clúster
    output_path = f"./data/procesados/07. reglas_apriori_cluster_{cluster_id}.csv"
    reglas.to_csv(output_path, index=False)
    print(f"✅ Reglas guardadas en {output_path}")
