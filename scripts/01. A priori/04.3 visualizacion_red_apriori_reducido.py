###################################################################################################################
###################################################################################################################
################################ 4- Visualización de reglas como red (a priori) ###################################
###################################################################################################################
######################################## Parte 3: Red de productos (reducido) #####################################
###################################################################################################################
###################################################################################################################

###################################################################################################################

# Este gráfico muestra una red reducida con las reglas más fuertes del análisis apriori,
# considerando solo aquellas con elevación (lift) > 3.2 y confianza > 0.2.
# Cada nodo representa un producto, y cada flecha una regla de asociación:
# si se compra el producto origen, es probable que se compre también el producto destino.
#
# Este enfoque reducido permite centrarse en las asociaciones más significativas, filtrando el ruido visual,
# y haciendo que las relaciones clave sean más fáciles de interpretar.

###################################################################################################################

# 1. Importar librerías
# 2. Cargar reglas de asociación traducidas
# 3. Filtrar reglas con elevación > 3.2 y confianza > 0.2
# 4. Convertir conjuntos a strings para visualización
# 5. Crear el grafo y exportar

###################################################################################################################


################################# 1. Importar librerías #################################

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

################################# 2. Cargar reglas de asociación traducidas #################################

reglas = pd.read_csv("././data/procesados/03T. reglas_apriori_traducido.csv")

################################# 3. Filtrar reglas con elevación > 3.2 y confianza > 0.2 #################################

reglas_red = reglas[
    (reglas["elevacion"] > 3.2) & 
    (reglas["confianza"] > 0.2)
].copy()

################################# 4. Convertir conjuntos a strings para visualización #################################

reglas_red["antecedentes"] = reglas_red["antecedentes"].str.strip("frozenset({})").str.replace("'", "")
reglas_red["consecuentes"] = reglas_red["consecuentes"].str.strip("frozenset({})").str.replace("'", "")

################################# 5. Crear el grafo y exportar #################################

G = nx.DiGraph()

# Añadir nodos y aristas
for _, fila in reglas_red.iterrows():
    G.add_edge(fila["antecedentes"], fila["consecuentes"], weight=fila["elevacion"])

# ================================
# NUEVO: Color distinto según rol
# ================================

antecedentes_set = set(reglas_red["antecedentes"])
consecuentes_set = set(reglas_red["consecuentes"])

node_colors = []
for nodo in G.nodes():
    if nodo in antecedentes_set and nodo in consecuentes_set:
        node_colors.append("#82E0AA")   # verde → ambos roles
    elif nodo in antecedentes_set:
        node_colors.append("#85C1E9")   # azul → antecedente
    else:
        node_colors.append("#F5B041")   # naranja → consecuente

# Tamaño de nodos según lift asociado
node_sizes = []
for nodo in G.nodes():
    lifts_ante = reglas_red[reglas_red["antecedentes"] == nodo]["elevacion"].tolist()
    lifts_cons = reglas_red[reglas_red["consecuentes"] == nodo]["elevacion"].tolist()
    lifts_total = lifts_ante + lifts_cons
    max_lift = max(lifts_total) if lifts_total else 1
    node_sizes.append(max_lift * 1000)

# Visualización
plt.figure(figsize=(12, 8))
pos = nx.kamada_kawai_layout(G)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=node_sizes,
    node_color=node_colors,   # ← ← ← ÚNICO CAMBIO EN EL DIBUJADO
    edgecolors="black",
    linewidths=0.5,
    font_size=10,
    font_color="#1C2833",
    font_family="sans-serif",
    arrows=True,
    arrowstyle="->",
    arrowsize=20,
    edge_color="#34495E",
    width=2,
    connectionstyle="arc3,rad=0.2"
)

plt.title("Red Reducida de Reglas de Asociación (elevación (lift) > 3.2, confianza > 0.2)", fontsize=14)
plt.axis("off")

plt.tight_layout()
plt.savefig("././output/01. A priori/03. gráfico_de_reglas_red_reducido.png")
plt.show()
print("✅ Red reducida de productos exportada a: output/03. gráfico_de_reglas_red_reducido.png")
