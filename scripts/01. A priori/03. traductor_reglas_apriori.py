###################################################################################################################
###################################################################################################################
############################## 3- Traductor de reglas de asociación (Apriori) ####################################
##################################### Traduce archivo de reglas al español #######################################
###################################################################################################################
###################################################################################################################


###################################################################################################################

# Este script traduce el archivo generado por Apriori al español, incluyendo:
# - Nombres de columnas como 'support', 'confidence', 'lift', etc.
# - Los contenidos dentro de los frozensets, es decir, los nombres de productos.
# Esto permite que los gráficos y análisis posteriores sean completamente legibles y presentables.

###################################################################################################################

# 1. Configuración del traductor
# 2. Cargar archivo original
# 3. Traducir nombres de columnas
# 4. Extraer todos los ítems únicos desde los frozensets
# 5. Crear diccionario de traducciones
# 6. Función para traducir frozensets
# 7. Aplicar traducción
# 8. Exportar archivo traducido


################################# 1. Configuración del traductor #################################

import pandas as pd
import re
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source="en", target="es")

def traducir_texto(texto: str) -> str:
    """
    Traduce una cadena del inglés al español.
    Si falla, devuelve el texto original.
    """
    try:
        return translator.translate(texto)
    except Exception:
        return texto  # fallback en caso de error de la API


################################# 2. Cargar el archivo original #################################

input_file = "././data/procesados/03. reglas_apriori.csv"             # nombre del archivo original
output_file = "././data/procesados/03T. reglas_apriori_traducido.csv" # nombre del archivo traducido

df = pd.read_csv(input_file)


################################# 3. Traducir nombres de columnas #################################

map_columnas = {
    "antecedents": "antecedentes",
    "consequents": "consecuentes",
    "antecedent support": "soporte antecedente",
    "consequent support": "soporte consecuente",
    "support": "soporte",
    "confidence": "confianza",
    "lift": "elevacion",
    "representativity": "representatividad",
    "leverage": "apalancamiento",
    "conviction": "conviccion",
    "zhangs_metric": "metrica_zhang",
    "jaccard": "jaccard",
    "certainty": "certeza",
    "kulczynski": "kulczynski",
}

df.rename(columns=map_columnas, inplace=True)


################################# 4. Extraer todos los textos de los frozenset #################################

columnas_sets = ["antecedentes", "consecuentes"]

todos_los_items = set()

patron = re.compile(r"'(.*?)'")  # captura lo que está entre comillas simples

for col in columnas_sets:
    if col in df.columns:
        for valor in df[col].dropna():
            valor_str = str(valor)
            items = patron.findall(valor_str)
            for it in items:
                todos_los_items.add(it)


################################# 5. Crear un diccionario de traducciones automáticas #################################

map_traducciones = {}

for item in todos_los_items:
    traduccion = traducir_texto(item)
    map_traducciones[item] = traduccion


################################# 6. Función para traducir el contenido de un frozenset #################################

def traducir_frozenset(celda):
    """
    Recibe algo como "frozenset({'Banana', 'Apple'})"
    y devuelve "frozenset({'Banana', 'Manzana'})", etc.,
    usando map_traducciones para cada elemento interno.
    """
    if not isinstance(celda, str):
        celda = str(celda)

    if "frozenset" not in celda:
        return celda

    items = patron.findall(celda)
    if not items:
        return celda

    items_traducidos = [
        map_traducciones.get(it, it) for it in items
    ]

    # reconstruir el string con el mismo formato
    interior = ", ".join(f"'{it}'" for it in items_traducidos)
    return f"frozenset({{{interior}}})"


################################# 7. Aplicar traducción a las columnas de sets #################################

for col in columnas_sets:
    if col in df.columns:
        df[col] = df[col].apply(traducir_frozenset)


################################# 8. Guardar el archivo traducido #################################

df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"✅ Archivo traducido correctamente → {output_file}")
