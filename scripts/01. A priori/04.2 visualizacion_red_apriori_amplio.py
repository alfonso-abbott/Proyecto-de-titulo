###################################################################################################################
###################################################################################################################
################################ 4- Visualización de reglas como red (a priori) ###################################
######################################## Parte 2: Red de productos ###########################################
###################################################################################################################
###################################################################################################################


###################################################################################################################

# Este gráfico construye una red de productos basándose en reglas de asociación con elevación > 1.5.
# Cada nodo representa un producto y cada flecha una regla: si se compra el producto origen,
# es probable que se compre también el producto destino.
#
# A diferencia del gráfico de dispersión, este permite visualizar directamente los productos asociados,
# facilitando la interpretación de patrones reales de compra y permitiendo construir estrategias concretas
# (recomendaciones personalizadas, combos frecuentes, cambios de layout en góndolas, etc.).

###################################################################################################################


# 1. Importar librerías
# 2. Cargar reglas de asociación traducidas
# 3. Filtrar reglas con mayor elevación
# 4. Convertir conjuntos a strings para visualización
# 5. Crear el grafo y exportar

###################################################################################################################


################################# 1. Importar librerías #################################


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


################################# 2. Cargar reglas de asociación traducidas #################################


reglas = pd.read_csv("././data/procesados/03T. reglas_apriori_traducido.csv")


################################# 3. Filtrar reglas con mayor elevación #################################
reglas_red = reglas[
    (reglas["elevacion"] > 1.5) & 
    (reglas["confianza"] > 0.2)
].copy()




################################# 4. Convertir conjuntos a strings para visualización #################################


reglas_red["antecedentes"] = reglas_red["antecedentes"].str.strip("frozenset({})").str.replace("'", "")
reglas_red["consecuentes"] = reglas_red["consecuentes"].str.strip("frozenset({})").str.replace("'", "")


################################# 5. Crear el grafo y exportar #################################


G = nx.DiGraph()

# Añadir nodos y aristas con peso según elevación
for _, fila in reglas_red.iterrows():
    G.add_edge(fila["antecedentes"], fila["consecuentes"], weight=fila["elevacion"])

# Visualización
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.95, iterations=20)

# Dibujar nodos y etiquetas
nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue", linewidths=0.5)
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray", width=1.5)
nx.draw_networkx_labels(G, pos, font_size=7, font_family="calibri")

# Título
plt.title("Red de Reglas de Asociación (lift > 1.5, confianza > 0.2)", fontsize=14)
plt.axis("off")

# Guardar gráfico
plt.tight_layout()
plt.savefig("././output/01. A priori/02. gráfico_de_reglas_red_amplio.png")
plt.show()

print("✅ Red de productos exportada a: output/02. gráfico_de_reglas_red_amplio.png")
