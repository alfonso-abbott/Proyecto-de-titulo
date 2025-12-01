###################################################################################################################
###################################################################################################################
########################## 02T. Visualización Top 5 reglas por clúster (Traducido) ################################
##################################### Cruce Apriori + K-means – Estilo Premium ###################################
###################################################################################################################
###################################################################################################################

################################# 1. Importar librerías #################################

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


################################# 2. Cargar archivo traducido #################################

ruta_resumen = "./data/procesados/08T. resumen_reglas_apriori_clusters_traducido.csv"
reglas = pd.read_csv(ruta_resumen)


################################# 3. Convertir frozenset a texto limpio #################################

def formatear_productos(fs):
    if isinstance(fs, str):
        fs = eval(fs)
    return ", ".join(fs)

reglas["antecedentes"] = reglas["antecedentes"].apply(formatear_productos)
reglas["consecuentes"] = reglas["consecuentes"].apply(formatear_productos)

reglas["regla"] = reglas["antecedentes"] + " → " + reglas["consecuentes"]


################################# 4. Filtrar reglas fuertes #################################

reglas_filtradas = reglas[
    (reglas["elevacion"] > 2.0) &
    (reglas["confianza"] > 0.2)
].copy()


################################# 5. Top 5 por clúster #################################

top_reglas = (
    reglas_filtradas
    .sort_values(["cluster", "elevacion"], ascending=[True, False])
    .groupby("cluster")
    .head(5)
)

# Ordenar para mejor lectura
top_reglas = top_reglas.sort_values(["cluster", "elevacion"], ascending=[True, False])


################################# 6. Crear gráfico profesional #################################

sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(15, 10))

# Paleta personalizada por clúster
palette = sns.color_palette("tab10", n_colors=top_reglas["cluster"].nunique())

# Dibujar barras horizontales
bars = ax.barh(
    top_reglas["regla"],
    top_reglas["elevacion"],
    color=[palette[c] for c in top_reglas["cluster"]],
    edgecolor="black",
    linewidth=0.8
)

# Etiquetas al lado derecho de cada barra
for i, row in top_reglas.iterrows():
    ax.text(
        row["elevacion"] + 0.10,              # desplazamiento a la derecha
        list(top_reglas.index).index(i),      # posición vertical
        f"conf: {row['confianza']:.2f} | supp: {row['soporte']:.2f}",
        va="center",
        fontsize=10,
        color="#2c3e50"
    )

# Título
ax.set_title(
    "Top 5 Reglas por Clúster \nElevación (lift) > 2.0 y Confianza > 0.2",
    fontsize=18,
    fontweight="bold",
    pad=20
)

# Etiquetas de ejes
ax.set_xlabel("Elevación (lift)", fontsize=14)
ax.set_ylabel("Reglas de Asociación", fontsize=13)

# Mejoras visuales
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=11)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.gca().invert_yaxis()

# Leyenda de clúster
clusters_unicos = sorted(top_reglas["cluster"].unique())
legend_handles = [
    plt.Line2D([0], [0], marker='s', color='w', label=f"Clúster {c}", 
               markerfacecolor=palette[c], markersize=10)
    for c in clusters_unicos
]
plt.legend(
    handles=legend_handles,
    title="Clúster",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()


################################# 7. Guardar gráfico #################################

output_path = "./output/04. Cruce A priori y K-means/01. Top reglas por clúster.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ Gráfico guardado en: {output_path}")
