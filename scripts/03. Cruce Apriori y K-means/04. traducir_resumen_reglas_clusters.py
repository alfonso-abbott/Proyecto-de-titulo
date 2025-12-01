###################################################################################################################
###################################################################################################################
############################## Traductor del resumen de reglas Apriori por clúster ################################
##################################### Traduce archivo de reglas al español ########################################
###################################################################################################################
###################################################################################################################

import pandas as pd
import re
from deep_translator import GoogleTranslator

###################################################################################################################
# 1. Configuración del traductor
###################################################################################################################

translator = GoogleTranslator(source="en", target="es")

def traducir_texto(texto: str) -> str:
    """
    Traduce una cadena del inglés al español.
    Si falla, devuelve el texto original.
    """
    try:
        return translator.translate(texto)
    except Exception:
        return texto  # fallback en caso de error en la API


###################################################################################################################
# 2. Cargar archivo original
###################################################################################################################

input_file = "./data/procesados/08. resumen_reglas_apriori_clusters.csv"
output_file = "./data/procesados/08T. resumen_reglas_apriori_clusters_traducido.csv"

df = pd.read_csv(input_file)


###################################################################################################################
# 3. Traducir nombres de columnas
###################################################################################################################

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
    "cluster": "cluster"
}

df.rename(columns=map_columnas, inplace=True)


###################################################################################################################
# 4. Extraer todos los textos de los frozenset
###################################################################################################################

columnas_sets = ["antecedentes", "consecuentes"]

todos_los_items = set()

# Extrae lo que está entre comillas simples dentro de los frozenset
patron = re.compile(r"'(.*?)'")

for col in columnas_sets:
    if col in df.columns:
        for valor in df[col].dropna():
            valor_str = str(valor)
            items = patron.findall(valor_str)
            for it in items:
                todos_los_items.add(it)


###################################################################################################################
# 5. Crear diccionario de traducciones automáticas
###################################################################################################################

map_traducciones = {}

for item in todos_los_items:
    traduccion = traducir_texto(item)
    map_traducciones[item] = traduccion


###################################################################################################################
# 6. Función para traducir frozensets
###################################################################################################################

def traducir_frozenset(celda):
    """
    Recibe algo como "frozenset({'Banana', 'Apple'})"
    y devuelve "frozenset({'Banana', 'Manzana'})"
    usando map_traducciones.
    """
    if not isinstance(celda, str):
        celda = str(celda)

    if "frozenset" not in celda:
        return celda

    items = patron.findall(celda)
    if not items:
        return celda

    items_traducidos = [map_traducciones.get(it, it) for it in items]

    interior = ", ".join(f"'{it}'" for it in items_traducidos)
    return f"frozenset({{{interior}}})"


###################################################################################################################
# 7. Aplicar traducción a columnas de sets
###################################################################################################################

for col in columnas_sets:
    if col in df.columns:
        df[col] = df[col].apply(traducir_frozenset)


###################################################################################################################
# 8. Guardar archivo traducido
###################################################################################################################

df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"✅ Archivo traducido correctamente → {output_file}")
