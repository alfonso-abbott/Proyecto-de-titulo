
# 1. Importar librerías
# 2. Cargar archivos
# 3. Unificación de datasets
# 4. Limpieza de datos (vamos con eso ahora)
# 5. Exportación de estructuras limpias



################################# 1. Importar librerías  #################################


import pandas as pd

# Rutas a los archivos
path = "./data/"


################################# # 2. Cargar archivos  #################################


orders = pd.read_csv(path + "orders.csv")
order_products_prior = pd.read_csv(path + "order_products__prior.csv")
order_products_train = pd.read_csv(path + "order_products__train.csv")
products = pd.read_csv(path + "products.csv")
departments = pd.read_csv(path + "departments.csv")
aisles = pd.read_csv(path + "aisles.csv")

# Mostrar primeras filas
print("📦 Orders:")
print(orders.head(), "\n")

print("🛒 Order Products (prior):")
print(order_products_prior.head(), "\n")

print("🛍️ Order Products (train):")
print(order_products_train.head(), "\n")

print("🧾 Products:")
print(products.head(), "\n")

print("🏢 Departments:")
print(departments.head(), "\n")

print("🧭 Aisles:")
print(aisles.head(), "\n")

# Mostrar shape (dimensiones)
print("🔢 Shapes:")
print(f"orders: {orders.shape}")
print(f"order_products_prior: {order_products_prior.shape}")
print(f"order_products_train: {order_products_train.shape}")
print(f"products: {products.shape}")
print(f"departments: {departments.shape}")
print(f"aisles: {aisles.shape}\n")

# Mostrar información general
print("ℹ️ Orders Info:")
orders.info()
print("\nℹ️ Products Info:")
products.info()



################################# 3. Unificación de datasets  #################################


# 🔗 Merge de información

# 1. Agregar nombres de productos
products_full = products.merge(aisles, on="aisle_id", how="left")
products_full = products_full.merge(departments, on="department_id", how="left")

# 2. Unir detalles de productos con pedidos
order_details = order_products_prior.merge(products_full, on="product_id", how="left")

# 3. Agregar información de orden
full_data = order_details.merge(orders, on="order_id", how="left")

# 🔎 Verificación inicial
print(full_data.head())
print("\n🔢 Dimensión final:", full_data.shape)



################################# 4. Limpieza de datos #################################


# Verificar valores nulos
print("\n🔍 Valores nulos por columna:")
print(full_data.isnull().sum())

# Eliminar filas con nombres de productos faltantes (por seguridad)
full_data = full_data.dropna(subset=["product_name"])

# Verificar duplicados
duplicados = full_data.duplicated().sum()
print(f"\n📛 Filas duplicadas: {duplicados}")

# Eliminar duplicados si existen
if duplicados > 0:
    full_data = full_data.drop_duplicates()

# Convertir ciertos campos a categorías para eficiencia
categorical_cols = ["product_name", "department", "aisle", "eval_set"]
for col in categorical_cols:
    full_data[col] = full_data[col].astype("category")

# 🔎 Confirmar limpieza
print("\n✅ Dataset limpio:")
print(full_data.info())
print("\n🧹 Dimensión final tras limpieza:", full_data.shape)



################### 5. Exportación de estructuras limpias ###################


# Carpeta de salida
output_path = "./data/processed/"

# Crear carpeta si no existe (opcional, si prefieres orden)
import os
os.makedirs(output_path, exist_ok=True)

### 1️⃣ Exportar dataset para Apriori ###
# Cada fila representa un producto comprado en una orden
# Columnas clave: order_id, product_name
apriori_df = full_data[["order_id", "product_name"]]
apriori_df.to_csv(output_path + "transacciones_apriori.csv", index=False)
print("\n📤 Exportado: transacciones_apriori.csv")

### 2️⃣ Exportar dataset para Clustering ###
# Dataset resumido por usuario
# Seleccionamos algunas columnas útiles para perfilar comportamiento
clustering_df = full_data[[
    "user_id", "order_id", "order_number", "order_dow",
    "order_hour_of_day", "days_since_prior_order", "department"
]]

# Podrías luego hacer un groupby por usuario si lo deseas
clustering_df.to_csv(output_path + "clientes_clustering.csv", index=False)
print("📤 Exportado: clientes_clustering.csv")
