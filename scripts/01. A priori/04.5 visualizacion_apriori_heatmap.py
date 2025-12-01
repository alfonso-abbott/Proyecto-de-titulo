###################################################################################################################
###################################################################################################################
######################## 4- Visualización de reglas como matriz de calor (a priori) ###############################
##################################### Parte 5: Matriz de calor de asociaciones ####################################
###################################################################################################################
###################################################################################################################

###################################################################################################################
# Este gráfico de matriz de calor muestra las asociaciones más fuertes entre productos (reglas Apriori),
# representadas como una tabla donde el eje X son los productos antecedente, y el eje Y los productos consecuente.
# El valor en cada celda representa el "lift" (elevación) de la regla.
#
# Se incluyen solo reglas con elevación > 3.2 y confianza > 0.2, para centrarse en asociaciones fuertes.
# Este tipo de visualización es útil para comparar rápidamente patrones entre múltiples combinaciones.
###################################################################################################################

# 1. Importar librerías
# 2. Cargar reglas traducidas
# 3. Filtrar por elevación > 3.2 y confianza > 0.2
# 4. Preparar DataFrame de matriz (antecedente vs. consecuente)
# 5. Visualizar como heatmap y exportar

###################################################################################################################


################################# 1. Importar librerías #################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


################################# 2. Cargar reglas traducidas #################################

reglas = pd.read_csv("././data/procesados/03T. reglas_apriori_traducido.csv")


################################# 3. Filtrar por elevación > 3.2 y confianza > 0.2 #################################

reglas_filtradas = reglas[
    (reglas["elevacion"] > 3.2) &
    (reglas["confianza"] > 0.2)
].copy()


################################# 4. Preparar DataFrame de matriz (antecedente vs. consecuente) #################################

# Limpiar etiquetas
reglas_filtradas["antecedentes"] = reglas_filtradas["antecedentes"].str.strip("frozenset({})").str.replace("'", "")
reglas_filtradas["consecuentes"] = reglas_filtradas["consecuentes"].str.strip("frozenset({})").str.replace("'", "")

# Crear tabla pivote: filas = antecedente, columnas = consecuente, valores = elevación
matriz = reglas_filtradas.pivot_table(
    index="antecedentes", 
    columns="consecuentes", 
    values="elevacion", 
    aggfunc="max"
).fillna(0)


################################# 5. Visualizar como heatmap y exportar #################################

plt.figure(figsize=(12, 9))
sns.set(font_scale=0.9)
sns.heatmap(
    matriz,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    linecolor="gray",
    square=True,
    cbar_kws={"label": "Elevación (lift)"}
)

plt.title("Matriz de Calor de Reglas de Asociación\n(lift > 3.2 y confianza > 0.2)", fontsize=14, pad=20)
plt.xlabel("Producto Consecuente")
plt.ylabel("Producto Antecedente")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Guardar imagen
plt.savefig("././output/01. A priori/05. apriori_matriz_de_calor.png", dpi=300)
plt.show()

print("✅ Matriz de calor exportada a: output/05. apriori_matriz_de_calor.png")
