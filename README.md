
---



---

# Proyecto de Título: Segmentación de Clientes y Patrones de Compra

Este repositorio implementa un flujo completo de minería de datos sobre el dataset de compras de supermercados en línea (Instacart). El proyecto integra **reglas de asociación (Apriori)**, **segmentación de clientes con K-Means**, un **cruce de ambos enfoques** y un **dashboard interactivo en Dash/Plotly** para explotar los resultados.

---

## 1. Arquitectura y organización

* `data/`: fuentes crudas (`datos/`) y salidas intermedias (`procesados/`).
* `scripts/`: pipeline modular en tres etapas:

  * `01. A priori/`: preprocesamiento, generación de reglas y visualizaciones base.
  * `02. K-means/`: validación, clustering de clientes y gráficos de comportamiento.
  * `03. Cruce Apriori y K-means/`: asociación de reglas por clúster y resúmenes.
* `output/`: gráficos y figuras exportadas por los scripts (codo, silueta, PCA, redes, barras, heatmaps, etc.).
* `app_dashboard.py` + `pages/` + `assets/`: dashboard multipágina con estilo glassmorphism.
* `notebooks/`: exploraciones adicionales (no usadas en la ejecución principal).
* `lanzar_dashboard.bat`: arranque rápido en Windows.

---

## 2. Flujo de datos y procesamiento

### 2.1 Ingesta y limpieza inicial

`01. A priori/01. preprocessing.py` carga los CSV de Instacart (`orders`, `order_products__prior/train`, `products`, `departments`, `aisles`), une catálogo y órdenes, elimina nulos/duplicados y tipifica variables categóricas.

### 2.2 Derivaciones para Apriori y clustering

El mismo script genera dos datasets clave:

* **Transacciones Apriori** (`data/procesados/01. transacciones_apriori.csv`): pares `order_id`–`product_name` para modelar co-compra.
* **Base de clientes** (`data/procesados/02. clientes_clustering.csv`): tabla a nivel pedido con usuario, temporalidad y departamento, usada luego por K-Means.

### 2.3 Minería de reglas (Apriori)

`01. A priori/02. analisis_apriori.py` filtra el **top 100 productos**, limita a **20k órdenes**, agrupa artículos por pedido, binariza con `TransactionEncoder` y ejecuta `apriori` con `min_support=0.005`.
Las reglas se derivan con `association_rules` (lift ≥ 1) y se guardan en `data/procesados/03. reglas_apriori.csv` para uso global.

### 2.4 Traducción y visualizaciones de Apriori

Los archivos `03. traductor_reglas_apriori.py` y `04.x visualizacion_apriori*.py` traducen nombres a español y generan gráficos (dispersión soporte–confianza, redes amplias/reducidas, barras top-lift, heatmap de elevación) en `output/01. A priori/` para el dashboard.

### 2.5 Validación y modelado K-Means

`02. K-means/01. validación.py` construye variables por cliente, escala, calcula matriz de correlación, método del codo e índice de silueta para k en un rango, exportando gráficos y resumen técnico a `output/02. K-means validacion/` para justificar k=5.

`02. K-means/02. clustering_clientes.py` genera el dataset final por cliente (promedios temporales + frecuencias por departamento), lo escala, aplica **K-Means (k=5)** y exporta:

* `data/procesados/04. clientes_clusterizados.csv` (completo)
* `data/procesados/05. clientes_clusterizados_reducido.csv`

También se genera el gráfico de distribución de clústeres en `output/03. K-means/`.

### 2.6 Cruce Apriori + K-Means

`03. Cruce Apriori y K-means/01. cruce_apriori_kmeans.py` vincula órdenes con `user_id`, clúster asignado y nombre de producto, produciendo transacciones etiquetadas por clúster (`data/procesados/06. transacciones_apriori_por_cluster.csv`).

`03. Cruce Apriori y K-means/02. reglas_apriori_por_cluster.py` toma esas transacciones, muestrea por clúster, vuelve a ejecutar Apriori y genera reglas específicas por segmento (`data/procesados/07. reglas_apriori_cluster_*.csv`).

Scripts posteriores combinan y traducen resultados:

* `03. unir_reglas_apriori_clusters.py`
* `04. traducir_resumen_reglas_clusters.py`
* `05. top_reglas_por_cluster.py`
* `06. red_reglas_por_cluster.py`

El resultado final es un **resumen traducido** en:
`data/procesados/08T. resumen_reglas_apriori_clusters_traducido.csv`.

---

## 3. Dashboard interactivo

El dashboard multipágina (`app_dashboard.py` + `pages/`) usa Dash con estilos en `assets/styles.css` para navegar entre:

* **Inicio**
* **K-Means**
* **Apriori**
* **Cruce**

### Principales páginas

#### **Inicio (`pages/home.py`)**

Carga tablas procesadas, incluye tarjetas de acceso con mini-gráficos, histograma de clústeres y dispersión del cruce.

#### **K-Means (`pages/kmeans_page.py`)**

Calcula PCA 2D, gráficos de distribución, hora y número de pedidos, preferencias por departamento y radar normalizado por clúster, con filtros interactivos.

#### **Apriori (`pages/apriori.py`)**

Incluye filtros de lift/confianza/producto, dispersión soporte–confianza, top reglas, heatmap de elevación y redes amplias/reducidas construidas con NetworkX.

#### **Cruce (`pages/cruce.py`)**

Permite seleccionar clúster y umbrales para ver barras de top reglas por segmento y su red asociada.

---

## 4. Artefactos generados

### 📁 CSV procesados (`data/procesados/`)

* transacciones para Apriori
* bases de clustering
* reglas globales
* reglas por clúster
* resúmenes traducidos

### 🖼 Gráficos (`output/`)

* validación de k (codo, silueta)
* distribución de clústeres
* PCA
* barras de comportamiento
* redes de reglas
* heatmaps
* top reglas por clúster

### 🖥 Dashboard

Visualizaciones interactivas sobre los CSV procesados.
**No recalcula modelos en tiempo real**: solo consume resultados ya generados.

---

## 5. Instalación y ejecución local

### 5.1 Requisitos

* Python 3.10+
* Paquetes listados en `requirements.txt` (Dash, Plotly, pandas, scikit-learn, mlxtend, seaborn, etc.)
* CSV de Instacart ubicados en `data/datos/` (ya incluidos)

### 5.2 Configuración

```bash
python -m venv .venv
source .venv/bin/activate       # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5.3 Ejecución del pipeline (opcional si ya existen los CSV)

```bash
python "scripts/01. A priori/01. preprocessing.py"
python "scripts/01. A priori/02. analisis_apriori.py"
python "scripts/02. K-means/01. validación.py"
python "scripts/02. K-means/02. clustering_clientes.py"
```

Luego, para generar reglas por clúster:

Ejecutar los scripts en `scripts/03. Cruce Apriori y K-means/` en orden numérico.

### 5.4 Ejecutar el dashboard

```bash
python app_dashboard.py
```

Abrir en el navegador:

```
http://127.0.0.1:8050/
```

En Windows también puedes usar:

```
lanzar_dashboard.bat
```

---

## 6. Lógica general de extremo a extremo

1. **Consolidación de datos**: unión de catálogo y órdenes, limpieza y exportación de dos vistas clave (transacciones y base de clientes).
2. **Apriori global**: muestreo, filtrado y minería de reglas; traducción y visualización.
3. **Segmentación K-Means**: validación de k, escalado, clustering y generación de perfiles.
4. **Cruce Apriori–K-Means**: recreación de transacciones por clúster y generación de reglas específicas por segmento.
5. **Visualización final**: dashboard multipágina que explora reglas, clústeres y patrones cruzados sin recalcular modelos.

---

