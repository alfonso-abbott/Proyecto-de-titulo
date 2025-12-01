###################################################################################################################
###################################################################################################################
############################## 4- Visualización de reglas de asociación (a priori) ################################
###################################################################################################################
######################################## Parte 1: Gráfico de dispersión ###########################################
###################################################################################################################
###################################################################################################################


###################################################################################################################

# Este gráfico permite visualizar e interpretar tres métricas clave de cada regla de asociación:
# - Eje X: Soporte (soporte) → cuán frecuente es la combinación en el total de órdenes.
# - Eje Y: Confianza (confianza) → probabilidad de que ocurra el consecuente si se da el antecedente.
# - Tamaño y color del punto: Elevación (elevacion) → fuerza de la relación (círculos grandes y brillantes indican reglas fuertes).

# Gracias a esto se pueden identificar fácilmente:
# - Reglas frecuentes (alto soporte)
# - Reglas confiables (alta probabilidad condicional)
# - Reglas valiosas (elevación alta → asociaciones no aleatorias)

# Aunque la mayoría de las reglas tienen soporte bajo (esperable en retail), algunas destacan por su confianza alta y elevación elevada,
# lo que las convierte en candidatas ideales para sistemas de recomendación o generación de bundles promocionales.

###################################################################################################################


# 1. Importar librerías
# 2. Cargar reglas generadas
# 3. Filtrar reglas relevantes
# 4. Gráfico de dispersión


################################# 1. Importar librerías #################################

import pandas as pd
import matplotlib.pyplot as plt


################################# 2. Cargar reglas generadas #################################

reglas = pd.read_csv("././data/procesados/03T. reglas_apriori_traducido.csv")


################################# 3. Filtrar reglas relevantes #################################

reglas_filtradas = reglas[reglas["elevacion"] > 1]


################################# 4. Gráfico de dispersión #################################

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    reglas_filtradas["soporte"],
    reglas_filtradas["confianza"],
    s=reglas_filtradas["elevacion"] * 20,  # Tamaño proporcional a elevación
    alpha=0.6,
    edgecolors="w",
    c=reglas_filtradas["elevacion"],
    cmap="viridis"
)

# Detalles visuales

plt.title("Reglas de Asociación: Soporte vs Confianza", fontsize=14)
plt.xlabel("Soporte")
plt.ylabel("Confianza")
cbar = plt.colorbar(scatter)
cbar.set_label("Elevación (lift)", rotation=270, labelpad=15)

# Guardar gráfico

plt.tight_layout()
plt.savefig("././output/01. A priori/01. apriori_reglas_de_asociacion.png")
plt.show()
print("✅ Gráfico de dispersión exportado a: output/01. apriori_reglas_de_asociacion.png")