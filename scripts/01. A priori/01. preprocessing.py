

###################################################################################################################
###################################################################################################################
############################################### 1- PREPROCESAMIENTO ##################################################
###################################################################################################################
###################################################################################################################



# 1. Importar librerías y definir ruta
# 2. Cargar archivos y mostrar información
# 3. Preprocesamiento: Unificación de datasets
# 4. Preprocesamiento: Limpieza de datos
# 5. Preprocesamiento: exportación de estructuras limpias



################################# 1. Importar librerías y definir ruta #################################


import pandas as pd

# Rutas a los archivos
# path = camino o ruta

path = "././data/datos/"


################################# # 2. Cargar archivos y mostrar información #################################

# aisles.csv: --> Lista de pasillos donde se agrupan productos
# departments.csv: --> Departamentos generales de productos
# order_products__prior.csv: --> Productos comprados en pedidos anteriores (para análisis)
# order_products__train.csv: --> Productos comprados en pedidos recientes (para entrenamiento)
# orders.csv: --> Registro de todas las órdenes de compra con usuario, tipo y momento
# products.csv: --> Catálogo completo de productos (product_id, product_name, etc.)


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

# Mostrar shape (dimensiones: filas y columnas)
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



################################# 3. Preprocesamiento: Unificación de datasets  #################################


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



################################# 4. Preprocesamiento: Limpieza de datos #################################


# 1. Verificar valores nulos
print("\n🔍 Valores nulos por columna:")
print(full_data.isnull().sum())

# 2. Eliminar filas con nombres de productos faltantes (por seguridad)
full_data = full_data.dropna(subset=["product_name"])

# 3. Verificar duplicados
duplicados = full_data.duplicated().sum()
print(f"\n📛 Filas duplicadas: {duplicados}")

# 4. Eliminar duplicados si existen
if duplicados > 0:
    full_data = full_data.drop_duplicates()

# 5. Convertir ciertos campos a categorías para eficiencia
categorical_cols = ["product_name", "department", "aisle", "eval_set"]
for col in categorical_cols:
    full_data[col] = full_data[col].astype("category")

# 6. Confirmar limpieza
print("\n✅ Dataset limpio:")
print(full_data.info())
print("\n🧹 Dimensión final tras limpieza:", full_data.shape)



################### 5. Preprocesamiento: exportación de estructuras limpias #################################


# Carpeta de salida
output_path = "././data/procesados/"

# Crear carpeta si no existe
import os
os.makedirs(output_path, exist_ok=True)


###################################################################################################################

######### 1️⃣ Exportar dataset para Apriori (Análisis de reglas de asociación) #########

# Cada fila representa un producto comprado en una orden
# Columnas clave: order_id, product_name

apriori_df = full_data[["order_id", "product_name"]]
apriori_df.to_csv(output_path + "01. transacciones_apriori.csv", index=False)
print("\n📤 Exportado: 01. transacciones_apriori.csv")

###################################################################################################################

######### 2️⃣ Exportar dataset para Clustering) Análisis de clustering de clientes) #########

# Dataset resumido por usuario
# Seleccionamos algunas columnas útiles para perfilar comportamiento

clustering_df = full_data[[
    "user_id", "order_id", "order_number", "order_dow",
    "order_hour_of_day", "days_since_prior_order", "department"
]]


######### Groupby por usuario #########

clustering_df.to_csv(output_path + "02. clientes_clustering.csv", index=False)
print("📤 Exportado: 02. clientes_clustering.csv")


###################################################################################################################



################### 6. Resultado al correr ###################


"""

$ "C:/Users/alfon/Desktop/Proyecto de título/.venv/Scripts/python.exe" "c:/Users/alfon/Desktop/Proyecto de título/scripts/01. preprocessing.py"
📦 Orders:
   order_id  user_id eval_set  order_number  order_dow  order_hour_of_day  days_since_prior_order
0   2539329        1    prior             1          2                  8                     NaN
1   2398795        1    prior             2          3                  7                    15.0
2    473747        1    prior             3          3                 12                    21.0
3   2254736        1    prior             4          4                  7                    29.0
4    431534        1    prior             5          4                 15                    28.0

🛒 Order Products (prior):
   order_id  product_id  add_to_cart_order  reordered
0         2       33120                  1          1
1         2       28985                  2          1
2         2        9327                  3          0
3         2       45918                  4          1
4         2       30035                  5          0

🛍️ Order Products (train):
   order_id  product_id  add_to_cart_order  reordered
0         1       49302                  1          1
1         1       11109                  2          1
2         1       10246                  3          0
3         1       49683                  4          0
4         1       43633                  5          1

🧾 Products:
   product_id                                       product_name  aisle_id  department_id
0           1                         Chocolate Sandwich Cookies        61             19
1           2                                   All-Seasons Salt       104             13
2           3               Robust Golden Unsweetened Oolong Tea        94              7
3           4  Smart Ones Classic Favorites Mini Rigatoni Wit...        38              1
4           5                          Green Chile Anytime Sauce         5             13

🏢 Departments:
   department_id department
0              1     frozen
1              2      other
2              3     bakery
3              4    produce
4              5    alcohol

🧭 Aisles:
   aisle_id                       aisle
0         1       prepared soups salads
1         2           specialty cheeses
2         3         energy granola bars
3         4               instant foods
4         5  marinades meat preparation

🔢 Shapes:
orders: (3421083, 7)
order_products_prior: (32434489, 4)
order_products_train: (1384617, 4)
products: (49688, 4)
departments: (21, 2)
aisles: (134, 2)

ℹ️ Orders Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3421083 entries, 0 to 3421082
Data columns (total 7 columns):
 #   Column                  Dtype
---  ------                  -----
 0   order_id                int64
 1   user_id                 int64
 2   eval_set                object
 3   order_number            int64
 4   order_dow               int64
 5   order_hour_of_day       int64
 6   days_since_prior_order  float64
dtypes: float64(1), int64(5), object(1)
memory usage: 182.7+ MB

ℹ️ Products Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 49688 entries, 0 to 49687
Data columns (total 4 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   product_id     49688 non-null  int64
 1   product_name   49688 non-null  object
 2   aisle_id       49688 non-null  int64
 3   department_id  49688 non-null  int64
dtypes: int64(3), object(1)
memory usage: 1.5+ MB
   order_id  product_id  add_to_cart_order  reordered  ... order_number  order_dow  order_hour_of_day days_since_prior_order
0         2       33120                  1          1  ...            3          5                  9                    8.0
1         2       28985                  2          1  ...            3          5                  9                    8.0
2         2        9327                  3          0  ...            3          5                  9                    8.0
3         2       45918                  4          1  ...            3          5                  9                    8.0
4         2       30035                  5          0  ...            3          5                  9                    8.0

[5 rows x 15 columns]

🔢 Dimensión final: (32434489, 15)

🔍 Valores nulos por columna:
order_id                        0
product_id                      0
add_to_cart_order               0
reordered                       0
product_name                    0
aisle_id                        0
department_id                   0
aisle                           0
department                      0
user_id                         0
eval_set                        0
order_number                    0
order_dow                       0
order_hour_of_day               0
days_since_prior_order    2078068
dtype: int64

📛 Filas duplicadas: 0

✅ Dataset limpio:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32434489 entries, 0 to 32434488
Data columns (total 15 columns):
 #   Column                  Dtype
---  ------                  -----
 0   order_id                int64
 1   product_id              int64
 2   add_to_cart_order       int64
 3   reordered               int64
 4   product_name            category
 5   aisle_id                int64
 6   department_id           int64
 7   aisle                   category
 8   department              category
 9   user_id                 int64
 10  eval_set                category
 11  order_number            int64
 12  order_dow               int64
 13  order_hour_of_day       int64
 14  days_since_prior_order  float64
dtypes: category(4), float64(1), int64(10)
memory usage: 2.9 GB
None

🧹 Dimensión final tras limpieza: (32434489, 15)

📤 Exportado: 01. transacciones_apriori.csv
📤 Exportado: 02. clientes_clustering.csv
(.venv) 

"""