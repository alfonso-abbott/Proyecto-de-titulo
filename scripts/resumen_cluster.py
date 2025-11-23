############################################
# TABLA DE RESUMEN POR CLÚSTER 📋
# Archivo: resumen_cluster.py
############################################

import pandas as pd
import os

# 📥 Cargar dataset de clientes clusterizados
df = pd.read_csv("./data/processed/clientes_clusterizados.csv")

# 📊 Calcular promedios por cluster
resumen = df.groupby("cluster").mean(numeric_only=True)

# 📋 Redondear para mayor claridad
resumen = resumen.round(2)

# 👁️ Mostrar en consola
print("📊 Tabla resumen por clúster:")
print(resumen)

# 💾 Exportar a CSV
os.makedirs("./output", exist_ok=True)
resumen.to_csv("./output/tabla_resumen_clusters.csv")
print("\n✅ Exportado como: output/tabla_resumen_clusters.csv")
