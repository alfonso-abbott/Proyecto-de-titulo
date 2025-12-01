######################################################################################################
# 🛒 Preferencias de compra - Departamento más comprado por clúster
# Archivo: scripts/02. K-means/05. Departamento más comprado por clúster.py
# Salida: output/03. K-means/05. departamento_top_por_cluster.png
#
# Este gráfico muestra el departamento más comprado por cada clúster de clientes.
# Para ello, se transforma el DataFrame de formato ancho a largo (melt), y se agrupa
# por clúster y departamento, seleccionando el de mayor número de compras.
######################################################################################################

# 1. 📦 Importación de librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 2. 🛠️ Configuración general
sns.set(style="whitegrid", font_scale=1.1)
os.makedirs("./output/03. K-means", exist_ok=True)

# 3. 📥 Carga del dataset
df = pd.read_csv("./data/procesados/04. clientes_clusterizados.csv")

# 4. 🔍 Detectar columnas de departamentos
departamentos = df.columns[6:-1]  # Excluye columnas generales y 'cluster'

# 5. 🔄 Transformar de ancho a largo
df_largo = df.melt(id_vars=["cluster"], value_vars=departamentos,
                   var_name="departamento", value_name="n_compras")

# 6. 🔝 Obtener departamento más comprado por clúster
df_top = df_largo.groupby(["cluster", "departamento"])["n_compras"].sum().reset_index()
df_top = df_top.sort_values(["cluster", "n_compras"], ascending=[True, False])
df_top_max = df_top.groupby("cluster").first().reset_index()

# 7. 🎨 Gráfico moderno y profesional
plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    data=df_top_max,
    x="cluster",
    y="n_compras",
    hue="departamento",
    palette="tab10",
    dodge=False,
    edgecolor="black"
)

# 8. ✏️ Etiquetas de valor
for i, row in df_top_max.iterrows():
    barplot.text(
        x=i, y=row["n_compras"] + 1,
        s=f'{row["n_compras"]:.0f}',
        ha='center', va='bottom', fontsize=10, weight='bold'
    )

# 9. 🧼 Detalles finales
plt.title("Departamento más comprado por clúster", fontsize=15, weight='bold')
plt.xlabel("Clúster")
plt.ylabel("Cantidad total de compras")
plt.legend(title="Departamento", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis="y", linestyle="--", linewidth=0.5)
plt.tight_layout()

# 10. 💾 Guardar gráfico
plt.savefig("./output/03. K-means/05.b Preferencias de compra. Departamento más comprado por clúster.png", dpi=300)
plt.close()
