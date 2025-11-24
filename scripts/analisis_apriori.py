

###################################################################################################################
###################################################################################################################
########################### 2- ANÁLISIS DE REGLAS DE ASOCIACIÓN (ANÁLISIS A PRIORI)🛍️ #############################
###################################################################################################################
###################################################################################################################


# Se trabaja con el dataset procesado "transacciones_apriori.csv" que se creó en el script de preprocesamiento.py
# El resultado será un archivo "reglas_apriori.csv" con las reglas generadas. En output/reglas_apriori.csv



# 1. Importar librerías y definir ruta
# 2. Cargar dataset procesado
# 3. Filtrar productos más frecuentes (top 100)
# 4. Limitar número de órdenes (20.000)
# 5. Agrupar productos por orden
# 6. Codificar transacciones
# 7. Generar conjuntos frecuentes con Apriori
# 8. Generar reglas de asociación
# 9. Mostrar reglas principales
# 10. Guardar resultados


####################################### CONTEXTO IMPORTANTE #######################################

# Al ejecutar apriori() con el dataset completo, aparece un error de memoria:
#   Unable to allocate 149. GiB for an array with shape (3214874, 49677)

# Eso significa que intenta generar una matriz booleana inmensa:
# - 3.2 millones de órdenes (filas)
# - 49.677 productos distintos (columnas)
# - Cada celda es tipo bool → 1 bit por celda

# Eso requiere más de 149 GB de RAM, por lo tanto es inviable en mi equipo actual (8 GB RAM).

# ✅ Solución adoptada (válida y eficiente):
# 1. Filtrar a los 100 productos más frecuentes
# 2. Limitar el análisis a 20.000 órdenes de compra

# Esto permite que el algoritmo apriori funcione correctamente en equipos con recursos limitados.
# Aunque se pierde algo de información, se capturan las asociaciones más relevantes y frecuentes.
# Esta es una práctica común en análisis de reglas de asociación cuando se trabaja con grandes volúmenes de datos.
# Además, las reglas generadas siguen siendo útiles para entender patrones de compra.
# Se documenta esta decisión en el código para claridad futura.

# #################################################################################################



################################# 1. Importar librerías y definir ruta #################################


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os

################################# 2. Cargar dataset procesado #################################


df_apriori = pd.read_csv("./data/processed/transacciones_apriori.csv")


################################# 3. Filtrar productos más frecuentes (top 100) #################################


productos_frecuentes = df_apriori["product_name"].value_counts().head(100).index.tolist()
df_filtrado = df_apriori[df_apriori["product_name"].isin(productos_frecuentes)]


################################# 4. Limitar número de órdenes (20.000) #################################


ordenes_muestras = df_filtrado["order_id"].unique()[:20000]
df_filtrado = df_filtrado[df_filtrado["order_id"].isin(ordenes_muestras)]


################################# 5. agrupar productos por orden #################################


transacciones = df_filtrado.groupby("order_id")["product_name"].apply(list).tolist()


################################# 6. Codificar transacciones #################################


te = TransactionEncoder()
te_ary = te.fit(transacciones).transform(transacciones)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)


################################# 7. Generar conjuntos frecuentes con Apriori #################################


# Se probó inicialmente con min_support=0.02, pero resultó ser muy alto.
# Se redujo a min_support=0.005 para captar combinaciones que aparecen al menos en el 0.5% de las órdenes.


frecuentes = apriori(df_encoded, min_support=0.005, use_colnames=True)


################################# 8. Generar reglas de asociación #################################


# Las reglas generadas deben tener un lift ≥ 1.0, es decir, que aporten valor real a la asociación.


reglas = association_rules(frecuentes, metric="lift", min_threshold=1.0)


################################# 9. Mostrar reglas principales #################################


print("\n📋 Reglas generadas:")
print(reglas[["antecedents", "consequents", "support", "confidence", "lift"]].head())


################################# 10. Guardar resultados #################################


os.makedirs("./output", exist_ok=True)
reglas.to_csv("./output/reglas_apriori.csv", index=False)
print("\n✅ Reglas de asociación exportadas a: output/reglas_apriori.csv")



################################# 11. Resultados #################################

# Se han generado reglas de asociación basadas en los 100 productos más frecuentes y 20.000 órdenes, 
# lo que permite identificar patrones de compra relevantes con un uso eficiente de memoria.

# Cada regla muestra una relación del tipo: "Si se compra A, entonces también se tiende a comprar B"

# Se evaluaron según métricas como:
# - support: frecuencia con que aparece la combinación en el total de transacciones.
# - confidence: probabilidad de que B se compre dado que se compró A.
# - lift: qué tan fuerte es la relación respecto a la aleatoriedad (>1 implica asociación positiva).

# Estas reglas pueden ser utilizadas para:
# 1- Recomendadores de productos
# 2- Optimización de layout en tiendas online
# 3- Estrategias de bundles o promociones

# El archivo `reglas_apriori.csv` contiene todas las reglas generadas para un análisis más detallado.


################### 12. Resultado al correr ###################


"""

$ "C:/Users/alfon/Desktop/Re creación proyecto de título/.venv/Scripts/python.exe" "c:/Users/alfon/Desktop/Re creación proyecto de título/scripts/analisis_apriori.py"

📋 Reglas generadas:
                  antecedents                 consequents  support  confidence      lift
0                    (Banana)    (100% Whole Wheat Bread)  0.00645    0.031571  1.193619
1    (100% Whole Wheat Bread)                    (Banana)  0.00645    0.243856  1.193619
2    (Bag of Organic Bananas)  (Apple Honeycrisp Organic)  0.00960    0.058164  1.604529
3  (Apple Honeycrisp Organic)    (Bag of Organic Bananas)  0.00960    0.264828  1.604529
4  (Apple Honeycrisp Organic)                    (Banana)  0.00765    0.211034  1.032964

✅ Reglas de asociación exportadas a: output/reglas_apriori.csv


"""