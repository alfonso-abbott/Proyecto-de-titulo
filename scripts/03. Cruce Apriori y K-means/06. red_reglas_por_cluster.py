###################################################################################################################
######################## Red Reducida por Clúster (08T traducido – versión mejorada con título y legibilidad) #####
###################################################################################################################

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

input_file = "./data/procesados/08T. resumen_reglas_apriori_clusters_traducido.csv"
output_dir = "./output/04. Cruce A priori y K-means"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file)

def convertir_frozenset(fs):
    if isinstance(fs, str):
        fs = eval(fs)
    return list(fs)

df["antecedentes"] = df["antecedentes"].apply(convertir_frozenset)
df["consecuentes"] = df["consecuentes"].apply(convertir_frozenset)

clusters = sorted(df["cluster"].unique())

for cluster_id in clusters:

    reglas = df[
        (df["cluster"] == cluster_id) &
        (df["elevacion"] > 2.0) &
        (df["confianza"] > 0.25)
    ].copy()

    if reglas.empty:
        print(f"Clúster {cluster_id} no tiene reglas fuertes.")
        continue

    reglas["antecedentes_str"] = reglas["antecedentes"].apply(lambda x: ", ".join(x))
    reglas["consecuentes_str"] = reglas["consecuentes"].apply(lambda x: ", ".join(x))

    G = nx.DiGraph()

    for _, fila in reglas.iterrows():
        for ant in fila["antecedentes"]:
            for cons in fila["consecuentes"]:
                G.add_edge(ant, cons, weight=fila["elevacion"])

    antecedentes_set = set([x for sub in reglas["antecedentes"] for x in sub])
    consecuentes_set = set([x for sub in reglas["consecuentes"] for x in sub])

    node_colors = []
    for nodo in G.nodes():
        if nodo in antecedentes_set and nodo in consecuentes_set:
            node_colors.append("#82E0AA")
        elif nodo in antecedentes_set:
            node_colors.append("#85C1E9")
        else:
            node_colors.append("#F5B041")

    node_sizes = []
    for nodo in G.nodes():
        lifts = []
        lifts += reglas[reglas["antecedentes_str"].str.contains(nodo, regex=False)]["elevacion"].tolist()
        lifts += reglas[reglas["consecuentes_str"].str.contains(nodo, regex=False)]["elevacion"].tolist()

        max_lift = max(lifts) if lifts else 1
        size = min(max_lift * 900, 5000)  # 🔥 límite para evitar monstruos gigantes
        node_sizes.append(size)

    # 🔥 Layout adaptativo
    if len(G.nodes()) <= 25:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

    # 🔥 Resolución más alta
    plt.figure(figsize=(16, 12))

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.7,
        arrows=True,
        arrowstyle="->",
        arrowsize=18,
        edge_color="#34495E",
        width=2,
        font_size=9,
        connectionstyle="arc3,rad=0.15"
    )

    # 🔥 Título siempre visible
    plt.suptitle(
        f"Red Reducida por clúster {cluster_id}\nElevación > 2.0 | Confianza > 0.25",
        fontsize=17,
        fontweight="bold",
        y=0.98
    )

    plt.axis("off")
    plt.subplots_adjust(top=0.92)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"03T. Red por clúster {cluster_id}.png")
    plt.savefig(output_path, dpi=350)
    plt.close()

    print(f"🚀 Red reducida del clúster {cluster_id} guardada en: {output_path}")
