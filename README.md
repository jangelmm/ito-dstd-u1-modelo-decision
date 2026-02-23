# Sistema de Soporte a la Decisión (DSS) - Modelo de Árbol de Decisión

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-latest-darkblue.svg)](https://pandas.pydata.org/)

Este repositorio contiene la implementación de un motor lógico basado en un modelo de **Árbol de Decisión (Decision Tree Classifier)**. Está diseñado como un Sistema de Soporte a la Decisión (DSS) para asistir a los niveles gerenciales y directivos en la estructuración y resolución de problemas de clasificación bajo condiciones de riesgo e incertidumbre.

El modelo utiliza Aprendizaje Automático Supervisado para extraer patrones estadísticos de bases de datos históricas (archivos CSV o diccionarios en código) y automatizar la evaluación probabilística mediante criterios como el **Índice de Gini** y la **Entropía**.

---

## Índice de Documentación del Proyecto

Para mantener el código limpio y la arquitectura comprensible, la documentación se ha dividido en dos áreas clave. **Por favor, lee estos documentos para entender el funcionamiento y fundamento del sistema:**

1. ⚙️ **[Manual de Usuario y Configuración](manual_usuario.md)**
   _Contiene las instrucciones detalladas elaboradas por el equipo para adaptar el script a cualquier problema de clasificación, modificar hiperparámetros (criterio, profundidad, muestras) y cargar nuevos datasets._
2. 🧮 **[Justificación Matemática y Lógica del Modelo](Logica_Modelo.md)**
   _Documento técnico que detalla el marco teórico, el algoritmo de partición recursiva Top-Down, y el desglose matemático de las fórmulas de evaluación de riesgo algorítmico._

---

## Tecnologías y Dependencias

El motor algorítmico está desarrollado en Python puro y utiliza el stack estándar de ciencia de datos:

- `pandas`: Para la manipulación, limpieza y codificación de los datos (One-Hot Encoding).
- `scikit-learn`: Para el entrenamiento del modelo (`DecisionTreeClassifier`) y las métricas de evaluación de precisión.
- `matplotlib`: Para la renderización y exportación visual del árbol de decisión.

---

## Instalación y Uso Rápido

### 1. Clonar el repositorio

```bash
git clone [https://github.com/tu-usuario/ito-dstd-u1-modelo-decision.git](https://github.com/tu-usuario/ito-dstd-u1-modelo-decision.git)

cd ito-dstd-u1-modelo-decision
```

### 2. Instalar requerimientos

Asegúrate de tener instaladas las librerías necesarias. Puedes instalarlas mediante `pip`:

```bash
pip install pandas scikit-learn matplotlib
```

### 3. Ejecutar el modelo

Para correr el script con los datos de prueba integrados, simplemente ejecuta:

```bash
python arbolDecision.py
```

### 4. Salidas del Sistema

Al ejecutarse, la terminal imprimirá:

- El Dataset codificado.
- La importancia matemática de los atributos.
- Las reglas del árbol en formato de texto.
- La predicción y probabilidad para un "Nuevo Caso" de prueba.
- Las métricas de evaluación (Accuracy y Classification Report).

Además, el sistema generará y guardará automáticamente un archivo gráfico llamado `arbol_decision.png` en el directorio raíz.

---

## 📂 Estructura del Repositorio

- `arbolDecision.py` — Script principal que contiene la lógica del negocio y el entrenamiento.
- `datos_csv.csv` — Archivo de ejemplo con datos estructurados para evaluación.
- `Logica_Modelo.md` — Documentación teórica y matemática del DSS.
- `manual_usuario.md` — Guía de configuración y adaptación de código.
- `arbol_decision.png` — Salida gráfica autogenerada del árbol.

---

_Proyecto desarrollado para la unidad 1 de Sistemas de Soporte a la Decisión._
