#############################################
# VISUALIZACIÓN DE REGLAS DE ASOCIACIÓN 📊
# Parte 1: Gráfico de dispersión
#############################################

import pandas as pd
import matplotlib.pyplot as plt

# 📥 Cargar reglas generadas
reglas = pd.read_csv("./output/reglas_apriori.csv")

# 🎯 Filtrar reglas relevantes (opcional)
reglas_filtradas = reglas[reglas["lift"] > 1]

# 📊 Gráfico de dispersión
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    reglas_filtradas["support"],
    reglas_filtradas["confidence"],
    s=reglas_filtradas["lift"] * 20,  # Tamaño proporcional al lift
    alpha=0.6,
    edgecolors="w",
    c=reglas_filtradas["lift"],
    cmap="viridis"
)

# 🏷️ Detalles visuales
plt.title("Reglas de Asociación: Soporte vs Confianza", fontsize=14)
plt.xlabel("Support (Soporte)")
plt.ylabel("Confidence (Confianza)")
cbar = plt.colorbar(scatter)
cbar.set_label("Lift", rotation=270, labelpad=15)

# 💾 Guardar gráfico
plt.tight_layout()
plt.savefig("./output/reglas_dispersion.png")
plt.show()

print("✅ Gráfico de dispersión exportado a: output/reglas_dispersion.png")
