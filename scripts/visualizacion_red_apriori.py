#############################################
# VISUALIZACIÓN DE REGLAS COMO RED 🕸️
# Parte 2: Red de productos
#############################################

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 📥 Cargar reglas de asociación
reglas = pd.read_csv("./output/reglas_apriori.csv")

# 🧽 Filtrar las reglas con mayor lift (opcional para simplificar la red)
reglas_red = reglas[reglas["lift"] > 1.5].copy()

# ⚙️ Convertir conjuntos a strings para visualización
reglas_red["antecedents"] = reglas_red["antecedents"].str.strip("frozenset({})").str.replace("'", "")
reglas_red["consequents"] = reglas_red["consequents"].str.strip("frozenset({})").str.replace("'", "")

# 🌐 Crear el grafo
G = nx.DiGraph()

# Añadir nodos y aristas
for _, fila in reglas_red.iterrows():
    G.add_edge(fila["antecedents"], fila["consequents"], weight=fila["lift"])

# 🎨 Visualización
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5, iterations=20)

# Dibujar nodos y etiquetas
nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue")
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray", width=1.5)
nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif")

# Título
plt.title("Red de Reglas de Asociación (Lift > 1.5)")
plt.axis("off")

# 💾 Guardar gráfico
plt.tight_layout()
plt.savefig("./output/reglas_red.png")
plt.show()

print("✅ Red de productos exportada a: output/reglas_red.png")
