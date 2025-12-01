###################################################################################################################
###################################################################################################################
########################## 4- Visualización de reglas como gráfico de barras (a priori) ##########################
######################################## Parte 4: Reglas con mayor elevación #####################################
###################################################################################################################
###################################################################################################################

###################################################################################################################
# Este gráfico de barras presenta las 10 reglas con mayor elevación (lift) del análisis Apriori.
# Permite comparar visualmente cuáles son las asociaciones más fuertes entre productos.
#
# Además del lift, se incluye confianza y soporte como etiquetas en cada barra, para comprender:
# - Qué tan fuerte es la asociación (lift)
# - Qué tan frecuente es (soporte)
# - Qué tan confiable es (confianza)
###################################################################################################################

# 1. Importar librerías
# 2. Cargar reglas traducidas
# 3. Filtrar por elevación > 3.2 y confianza > 0.2
# 4. Seleccionar top 10 por elevación
# 5. Crear gráfico de barras estilizado y exportar

###################################################################################################################


################################# 1. Importar librerías #################################

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


################################# 2. Cargar reglas traducidas #################################

reglas = pd.read_csv("././data/procesados/03T. reglas_apriori_traducido.csv")


################################# 3. Filtrar por elevación > 3.2 y confianza > 0.2 #################################

reglas_filtradas = reglas[
    (reglas["elevacion"] > 3.2) & 
    (reglas["confianza"] > 0.2)
].copy()


################################# 4. Seleccionar top 10 por elevación #################################

reglas_top10 = reglas_filtradas.sort_values("elevacion", ascending=False).head(10)

# Crear una columna de texto amigable para el gráfico
reglas_top10["regla"] = (
    reglas_top10["antecedentes"].str.strip("frozenset({})").str.replace("'", "") +
    " → " +
    reglas_top10["consecuentes"].str.strip("frozenset({})").str.replace("'", "")
)


################################# 5. Crear gráfico de barras estilizado y exportar #################################

# Estilo visual
import seaborn as sns
sns.set_style("whitegrid")  # Puedes usar: "white", "dark", "whitegrid", "darkgrid", "ticks"

fig, ax = plt.subplots(figsize=(14, 8))

# Gráfico de barras horizontal
bars = ax.barh(reglas_top10["regla"], reglas_top10["elevacion"], color="#db34bf", edgecolor="black")

# Agregar etiquetas con confianza y soporte a la derecha de cada barra
for i, (conf, supp) in enumerate(zip(reglas_top10["confianza"], reglas_top10["soporte"])):
    ax.text(
        reglas_top10["elevacion"].iloc[i] + 0.15, i,
        f"conf: {conf:.2f} | supp: {supp:.2f}",
        va="center", ha="left", fontsize=10, color="#2c3e50"
    )

# Título y etiquetas
ax.set_title("Top 10 Reglas con Mayor Elevación\n(lift > 3.2 y confianza > 0.2)", fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel("Elevación (lift)", fontsize=13)
ax.set_ylabel("Reglas de asociación", fontsize=13)

# Ajustes estéticos
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=11)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.gca().invert_yaxis()
plt.tight_layout()

# Exportar gráfico
plt.savefig("././output/01. A priori/04. apriori_grafico_de_barras.png", dpi=300)
plt.show()

print("✅ Gráfico de barras estilizado exportado a: output/04. apriori_grafico_de_barras.png")
