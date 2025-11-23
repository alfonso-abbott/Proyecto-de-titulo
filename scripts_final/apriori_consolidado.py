##########################################
# CONSOLIDADO - ANÁLISIS DE REGLAS APRIORI 🧠📦
# Archivo: apriori_consolidado.py
##########################################

# 1. Librerías necesarias
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import networkx as nx

# 2. Cargar dataset procesado
df_apriori = pd.read_csv("./data/processed/transacciones_apriori.csv")

# 3. Agrupar productos por orden
transacciones = df_apriori.groupby("order_id")["product_name"].apply(list)
lista_transacciones = transacciones.tolist()

# 4. Codificar datos para Apriori
te = TransactionEncoder()
te_ary = te.fit(lista_transacciones).transform(lista_transacciones)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 5. Ejecutar Apriori
frecuentes = apriori(df_encoded, min_support=0.005, use_colnames=True)
reglas = association_rules(frecuentes, metric="lift", min_threshold=1.2)

# 6. Exportar reglas
reglas.to_csv("./output/reglas_apriori.csv", index=False)
print("✅ Reglas exportadas: output/reglas_apriori.csv")

# 7. Gráfico 1: Soporte vs Confianza con lift como tamaño y color
plt.figure(figsize=(12,6))
scatter = plt.scatter(
    reglas["support"],
    reglas["confidence"],
    s=reglas["lift"]*10,
    c=reglas["lift"],
    cmap='viridis',
    alpha=0.7,
    edgecolors='k'
)
plt.colorbar(scatter, label='Lift')
plt.title("Reglas de Asociación: Soporte vs Confianza")
plt.xlabel("Support (Soporte)")
plt.ylabel("Confidence (Confianza)")
plt.tight_layout()
plt.savefig("./output/reglas_apriori_dispersion.png")
plt.close()
print("📊 Gráfico de dispersión guardado: output/reglas_apriori_dispersion.png")

# 8. Gráfico 2: Red de asociaciones
# Filtrar reglas fuertes
reglas_red = reglas[reglas["lift"] > 1.5]

# Crear red
G = nx.DiGraph()

for _, row in reglas_red.iterrows():
    antecedente = ', '.join(list(row['antecedents']))
    consecuente = ', '.join(list(row['consequents']))
    G.add_edge(antecedente, consecuente, weight=row['lift'])

# Dibujar red
plt.figure(figsize=(14,8))
pos = nx.spring_layout(G, k=0.5)
nx.draw(
    G, pos,
    with_labels=True,
    node_size=1500,
    node_color="skyblue",
    font_size=8,
    font_weight="bold",
    arrows=True
)
plt.title("Red de Reglas de Asociación (Lift > 1.5)")
plt.tight_layout()
plt.savefig("./output/reglas_apriori_red.png")
plt.close()
print("🌐 Gráfico de red guardado: output/reglas_apriori_red.png")
