================================================================================
MANUAL DE USUARIO
Sistema de Soporte a la Decisión - Árbol de Decisión
Unidad 1: Modelo de Decisión
================================================================================

## ÍNDICE

1. Introducción
2. Requisitos del Sistema
3. Estructura del Proyecto
4. Archivos del Proyecto
5. ¿Cómo Funciona el Sistema?
6. Cómo Ejecutar el Programa
7. Guía Paso a Paso del Flujo de Ejecución
   7.1 Paso 1 – Selección del Criterio de División
   7.2 Paso 2 – Selección de la Profundidad Máxima del Árbol
   7.3 Paso 3 – Visualización del Dataset
   7.4 Paso 4 – Features Codificadas
   7.5 Paso 5 – Información del Modelo Entrenado
   7.6 Paso 6 – Árbol de Decisión Visual
   7.7 Paso 7 – Predicción del Nuevo Caso
   7.8 Paso 8 – Evaluación del Modelo
   7.9 Paso 9 – Reglas del Árbol en Texto
   7.10 Paso 10 – Simulador Interactivo (What-If)
8. Cómo Adaptar el Sistema a Otro Problema
   8.1 Sección 1: Cambiar el Dataset
   8.2 Sección 2: Cambiar la Columna Objetivo (Target)
   8.3 Sección 3: Cambiar el Nuevo Caso a Predecir
9. Descripción de los Parámetros de Configuración
10. Interpretación de los Resultados
11. Preguntas Frecuentes (FAQ)
12. Glosario de Términos

================================================================================

1. # INTRODUCCIÓN

Este sistema es una herramienta de Soporte a la Decisión (DSS, por sus siglas
en inglés: Decision Support System) que implementa un Árbol de Decisión para
ayudar a los directivos y analistas a clasificar casos y tomar decisiones bajo
condiciones de incertidumbre, basándose en datos históricos.

El caso de uso predeterminado del sistema es un diagnóstico médico simplificado:
a partir de síntomas de un paciente (Fiebre, Tos, Dolor corporal y si el
paciente tiene más de 50 años), el sistema predice si dicho paciente tiene o no
una enfermedad.

Sin embargo, el sistema está diseñado para ser completamente reutilizable y
adaptable a cualquier problema de clasificación binaria o multiclase, sin
necesidad de modificar la lógica del programa; solo se requiere cambiar tres
secciones claramente marcadas en el código.

Características principales del sistema:

- Entrenamiento automático del modelo con datos históricos.
- Visualización gráfica del árbol de decisión (imagen PNG).
- Predicción de nuevos casos con probabilidades asociadas.
- Evaluación del rendimiento del modelo (accuracy y reporte de clasificación).
- Presentación de reglas de decisión en formato de texto legible.
- Simulador interactivo "What-If" para explorar escenarios alternativos.
- Soporte para dos criterios matemáticos: Gini y Entropía.

================================================================================ 2. REQUISITOS DEL SISTEMA
================================================================================

Para ejecutar este sistema correctamente, se necesita tener instalado:

Software: - Python 3.8 o superior - Las siguientes librerías de Python:
_ pandas (manejo de datos y DataFrames)
_ scikit-learn (algoritmo de árbol de decisión) \* matplotlib (generación de la imagen del árbol)

Instalación de librerías (ejecutar en la terminal):
pip install pandas scikit-learn matplotlib

================================================================================ 3. ESTRUCTURA DEL PROYECTO
================================================================================

proyectoAD/
└── ito-dstd-u1-modelo-decision/
├── arbolDecision.py <- Script principal del sistema (EJECUTAR ESTE)
├── datos_csv.csv <- Dataset de ejemplo en formato CSV
├── arbol_decision.png <- Imagen del árbol (se genera al ejecutar)
├── Logica_Modelo.md <- Documentación técnica del modelo
├── README.md <- Información general del proyecto
└── manual_usuario_md <- Este manual

================================================================================ 4. ARCHIVOS DEL PROYECTO
================================================================================

arbolDecision.py

---

Es el programa principal, contiene toda la lógica del sistema: carga de datos,
entrenamiento del modelo, predicción, evaluación, visualización y el simulador
interactivo. Es el único archivo que se necesita ejecutar.

datos_csv.csv

---

Dataset de ejemplo en formato CSV con 8 pacientes y 5 columnas: - Fiebre : Nivel de fiebre del paciente (Alta / Baja) - Tos : Si el paciente tiene tos (si / no) - Dolor : Si el paciente tiene dolor corporal (si / no) - EdadMayor50 : Si el paciente tiene más de 50 años (si / no) - Enfermedad : Resultado (si / no) — esta es la columna a predecir

arbol_decision.png

---

Imagen en formato PNG del árbol de decisión entrenado. Se genera o actualiza
automáticamente cada vez que se ejecuta el programa.

Logica_Modelo.md

---

Documento técnico que explica el fundamento matemático del modelo, incluyendo:
la anatomía del árbol, el proceso de partición recursiva, las fórmulas del
índice Gini y la Entropía, y la interpretación de resultados.

README.md

---

Descripción general del proyecto, instrucciones de adaptación y configuración.

================================================================================ 5. ¿CÓMO FUNCIONA EL SISTEMA?
================================================================================

El sistema sigue el siguiente flujo de trabajo al ejecutarse:

FASE 1 – CONFIGURACIÓN INTERACTIVA
El usuario responde dos preguntas en consola:
a) Qué criterio matemático usar para entrenar el árbol (Gini o Entropía).
b) Cuál será la profundidad máxima del árbol (o sin límite).

FASE 2 – CARGA Y PREPARACIÓN DE DATOS
El programa lee el dataset (desde CSV o desde el diccionario interno),
separa las columnas de entrada (features) de la columna objetivo (target)
y codifica las variables categóricas (texto) a formato numérico para que
el algoritmo pueda procesarlas.

FASE 3 – ENTRENAMIENTO DEL MODELO
Con los datos preparados, el algoritmo DecisionTreeClassifier de la
librería scikit-learn construye el árbol de decisión. Durante este proceso,
el algoritmo analiza todas las divisiones posibles de los datos y selecciona
la que mejor separa las clases según el criterio elegido (Gini o Entropía),
repitiendo este proceso recursivamente hasta cumplir alguna condición de
parada.

FASE 4 – PRESENTACIÓN DE RESULTADOS
El sistema muestra en consola: - El dataset original completo. - Las features codificadas y el target. - Información del modelo entrenado (profundidad, hojas, importancia
de atributos). - La imagen del árbol de decisión (ventana gráfica + archivo PNG). - La predicción del nuevo caso predefinido en el código. - La evaluación del modelo (accuracy y reporte de clasificación). - Las reglas del árbol en texto legible.

FASE 5 – SIMULADOR WHAT-IF INTERACTIVO
Al finalizar el análisis, el sistema ofrece un simulador donde el usuario
puede modificar los valores del caso a predecir (uno a la vez) y ver en
tiempo real cómo cambia la decisión del árbol.

================================================================================ 6. CÓMO EJECUTAR EL PROGRAMA
================================================================================

Método 1 – Desde la terminal / consola: 1. Abra una terminal 2. Navegue a la carpeta del proyecto:
cd "C:\Users\PC\Desktop\proyectoAD\ito-dstd-u1-modelo-decision" 3. Ejecute el script:
python arbolDecision.py

Método 2 – Desde un IDE 1. Abra el archivo arbolDecision.py en su IDE. 2. Asegúrese de que la terminal integrada esté posicionada en la carpeta
del proyecto (donde también se encuentra datos_csv.csv). 3. Presione el botón "Run" o use el atajo correspondiente (F5 en VS Code).

IMPORTANTE: - El archivo datos_csv.csv debe estar en la MISMA CARPETA que arbolDecision.py
para que el programa lo encuentre correctamente. - El archivo arbol_decision.png se guardará también en esa misma carpeta.

================================================================================ 7. GUÍA PASO A PASO DEL FLUJO DE EJECUCIÓN
================================================================================

Al ejecutar el programa, el sistema irá mostrando información y solicitando
entradas del usuario en el siguiente orden:

---

## 7.1 PASO 1 – SELECCIÓN DEL CRITERIO DE DIVISIÓN

El programa mostrará el siguiente mensaje y esperará su respuesta:

Seleccione el criterio de decision (gini / entropy):

Opciones válidas:
gini → Usa el Índice de Impureza de Gini. Es el criterio más común y
rápido. Mide la probabilidad de clasificar incorrectamente un
elemento al azar.
entropy → Usa la Entropía (Ganancia de Información). Basado en la Teoría
de la Información. Puede generar árboles ligeramente distintos.

Ejemplo de entrada:
gini

Si escribe algo distinto, el sistema mostrará un mensaje de error y volverá
a pedir la entrada hasta que sea válida.

---

## 7.2 PASO 2 – SELECCIÓN DE LA PROFUNDIDAD MÁXIMA DEL ÁRBOL

A continuación, el programa preguntará:

Ingrese la profundidad maxima del arbol (Enter para sin limite):

Opciones: - Presione ENTER sin escribir nada: el árbol crecerá sin límite hasta que
los nodos sean completamente puros. Esto maximiza la precisión sobre los
datos de entrenamiento pero puede llevar al sobreajuste (overfitting). - Escriba un número entero positivo (ej: 2, 3, 4): limita la profundidad
del árbol al valor indicado. Árboles más cortos son más simples y
generalizables, pero pueden ser menos precisos.

Recomendación: - Para datasets pequeños (como el ejemplo de 8 filas): usar sin límite o
una profundidad de 3-4. - Para datasets grandes: limitar la profundidad entre 3 y 10 para evitar
el sobreajuste.

Ejemplo de entrada:
3

---

## 7.3 PASO 3 – VISUALIZACIÓN DEL DATASET

El sistema mostrará el dataset completo en formato de tabla:

=======================================================
DATASET
=======================================================
Fiebre Tos Dolor EdadMayor50 Enfermedad
0 Alta Si Si Si Si
1 Alta Si No No Si
...

Total filas: 8

Esto permite verificar que los datos se cargaron correctamente.

---

## 7.4 PASO 4 – FEATURES CODIFICADAS

El sistema muestra cómo quedan los datos después de la codificación numérica:

=======================================================
FEATURES CODIFICADAS
=======================================================
Fiebre_Si Tos_Si Dolor_Si EdadMayor50_Si
0 1 1 1 1
1 1 1 0 0
...

Las variables de texto (Si/No, Alta/Baja) se convierten en columnas de 0 y 1
mediante la técnica "one-hot encoding" para que el algoritmo pueda procesarlas.

---

## 7.5 PASO 5 – INFORMACIÓN DEL MODELO ENTRENADO

El sistema muestra las características del árbol generado:

=======================================================
MODELO ENTRENADO
=======================================================
Profundidad: 3
Num. hojas: 4
Criterio: gini
Max depth: None

IMPORTANCIA DE ATRIBUTOS
=======================================================
Fiebre_Si 0.4688 ##################
Dolor_Si 0.2813 ###########
Tos_Si 0.2500 ##########
EdadMayor50_Si 0.0000

La "Importancia de Atributos" indica qué tan relevante es cada variable para
que el modelo tome decisiones. Un valor cercano a 1.0 significa que ese
atributo es muy influyente; un valor de 0.0 significa que el árbol no lo usó.

---

## 7.6 PASO 6 – ÁRBOL DE DECISIÓN VISUAL

El programa abrirá una ventana gráfica mostrando el árbol de decisión en forma
de diagrama. Además, guardará la imagen como:
arbol_decision.png

Interpretación del diagrama: - Cada caja (nodo) contiene una condición de decisión. - Las ramas "True/False" o "Si/No" indican las dos posibles rutas. - Los nodos de colores más intensos representan mayor pureza (clases más
claramente separadas). - Los nodos hoja (sin hijos) muestran la predicción final: la clase
mayoritaria en ese grupo de datos.

Cierre la ventana gráfica para que el programa continúe con los siguientes
pasos.

---

## 7.7 PASO 7 – PREDICCIÓN DEL NUEVO CASO

El sistema muestra la predicción para el caso predefinido en el código
(Sección 3 de arbolDecision.py):

=======================================================
PREDICCION - NUEVO CASO
=======================================================
Valores de entrada:
Fiebre: Alta
Tos: Si
Dolor: No
EdadMayor50: No

    Clase predicha: Si

    P(No): 0.0000
    P(Si): 1.0000

Interpretación: - "Clase predicha" es la recomendación directa del sistema. - "P(No)" y "P(Si)" son las probabilidades de cada clase basadas en los
datos históricos que terminaron en la misma hoja del árbol. - Una probabilidad de 1.0000 (100%) indica un nodo completamente puro. - Una probabilidad de 0.75 indica un riesgo del 25% de que la predicción
sea incorrecta, lo que el directivo debe considerar al tomar su decisión.

---

## 7.8 PASO 8 – EVALUACIÓN DEL MODELO

El sistema evalúa el rendimiento general del modelo con el conjunto de datos:

=======================================================
EVALUACION (ENTRENAMIENTO (referencia))
=======================================================
Accuracy: 1.0000

                  precision    recall  f1-score   support
              No       1.00      1.00      1.00         5
              Si       1.00      1.00      1.00         3

Métricas: - Accuracy: Porcentaje de casos clasificados correctamente. 1.0000 = 100%. - Precision: De todos los casos que el modelo predijo como "Si", ¿cuántos
realmente lo eran? - Recall: De todos los casos que realmente son "Si", ¿cuántos detectó el
modelo? - F1-Score: Promedio armónico entre precision y recall. Útil cuando las
clases están desbalanceadas. - Support: Número de muestras reales de cada clase en el conjunto evaluado.

NOTA: Una accuracy de 1.0 (100%) sobre el mismo conjunto de entrenamiento
es esperada cuando no hay límite de profundidad y el dataset es pequeño.
Esto no garantiza buen rendimiento con datos nuevos.

---

## 7.9 PASO 9 – REGLAS DEL ÁRBOL EN TEXTO

El sistema imprime las reglas de decisión del árbol en formato de texto:

=======================================================
REGLAS DEL ARBOL (texto)
=======================================================
|--- Fiebre_Si <= 0.50
| |--- class: No
|--- Fiebre_Si > 0.50
| |--- Dolor_Si <= 0.50
| | |--- Tos_Si <= 0.50
| | | |--- class: No
| | |--- Tos_Si > 0.50
| | | |--- class: Si
| |--- Dolor_Si > 0.50
| | |--- class: Si

Interpretación: - Cada línea con "---" es una condición de decisión. - Las condiciones se evalúan de arriba hacia abajo. - "class:" al final de una rama indica la predicción (hoja del árbol). - Las sangrías representan la jerarquía: más sangría = más profundidad.

Ejemplo de lectura:
Si Fiebre_Si = 0 (es decir, Fiebre = Baja):
→ La predicción es: No (no tiene la enfermedad).
Si Fiebre_Si = 1 (es decir, Fiebre = Alta) Y Dolor_Si = 0 (no tiene dolor)
Y Tos_Si = 0 (no tiene tos):
→ La predicción es: No.

---

## 7.10 PASO 10 – SIMULADOR INTERACTIVO (WHAT-IF)

Tras mostrar todos los resultados, el sistema preguntará:

¿Desea cambiar un parámetro del cliente para probar diferentes escenarios
interactivos? (si/no):

Si responde "si", el simulador se activará y mostrará:

    -- Simulador interactivo de decisiones --
    =======================================================
    Parámetros actuales del cliente:
      Fiebre: Alta
      Tos: Si
      Dolor: No
      EdadMayor50: No

    --> Decisión del Árbol: SI <--

    ¿Qué parámetro desea modificar?
      1. Fiebre
      2. Tos
      3. Dolor
      4. EdadMayor50
      5. Salir y finalizar programa

    Seleccione el número del parámetro:

Al seleccionar un parámetro (ej: escribir "1" para Fiebre), el sistema
mostrará las opciones posibles:

    Opciones posibles para 'Fiebre':
      1. Alta
      2. Baja
      3. Cancelar y volver al menú principal

    Seleccione el número del nuevo valor:

Una vez seleccionado el nuevo valor, el simulador actualizará el parámetro
y mostrará de inmediato la nueva decisión del árbol. Esto permite explorar
preguntas del tipo:
"¿Qué pasaría si el paciente tuviera fiebre Baja en lugar de Alta?"
"¿Cambiaría la predicción si el paciente fuera mayor de 50 años?"

Para salir del simulador, seleccione la opción "Salir y finalizar programa".

Si responde "no" a la pregunta inicial, el programa terminará de inmediato.

================================================================================ 8. CÓMO ADAPTAR EL SISTEMA A OTRO PROBLEMA
================================================================================

El script arbolDecision.py puede adaptarse a cualquier problema de clasificación
modificando únicamente tres secciones claramente marcadas con el comentario:

> > > MODIFICAR AQUI <<<

---

## 8.1 SECCIÓN 1: CAMBIAR EL DATASET

Ubicación en el código: líneas ~148-173

Opción A – Cargar desde un archivo CSV (recomendado para datasets grandes):
Descomente la línea:
df = pd.read_csv("tu_archivo.csv")
Y asegúrese de que el archivo CSV esté en la misma carpeta que el script.

Opción B – Definir los datos directamente en el código:
Modifique el diccionario "data" con sus propias columnas y valores.

    Ejemplo para un problema de aprobación de crédito:
      data = {
          "Ingreso":       ["Alto", "Alto", "Bajo", "Bajo", "Medio"],
          "HistorialPago": ["Bueno", "Malo", "Bueno", "Malo", "Bueno"],
          "MontoSolicitado": ["Bajo", "Alto", "Bajo", "Alto", "Medio"],
          "Aprobado":      ["Si",   "No",   "Si",   "No",   "Si"],
      }

    Reglas para el dataset:
      - Cada clave del diccionario representa una columna.
      - Todas las listas deben tener la misma cantidad de elementos.
      - Las columnas pueden tener valores de texto (categóricos) o números.
      - La columna objetivo (lo que se quiere predecir) puede tener cualquier
        nombre, pero debe especificarse en la Sección 2.

---

## 8.2 SECCIÓN 2: CAMBIAR LA COLUMNA OBJETIVO (TARGET)

Ubicación en el código: líneas ~177-183

Cambie el valor de COLUMNA_TARGET para que coincida exactamente con el nombre
de la columna que desea predecir.

Ejemplo para el problema de crédito:
COLUMNA_TARGET = "Aprobado"

---

## 8.3 SECCIÓN 3: CAMBIAR EL NUEVO CASO A PREDECIR

Ubicación en el código: líneas ~187-199

Cambie el diccionario "nuevo_caso" con los valores del caso concreto que
desea clasificar. Las claves deben ser exactamente las mismas columnas que
tiene el dataset, excepto la columna objetivo (target).

Ejemplo para el problema de crédito:
nuevo_caso = {
"Ingreso": ["Medio"],
"HistorialPago": ["Bueno"],
"MontoSolicitado": ["Alto"],
}

================================================================================ 9. DESCRIPCIÓN DE LOS PARÁMETROS DE CONFIGURACIÓN
================================================================================

Los siguientes parámetros se pueden ajustar al inicio del script para modificar
el comportamiento del modelo (líneas 26-59 aproximadamente):

CRITERIO (string)
Valor: "gini" o "entropy"
Descripción: Criterio matemático que el algoritmo usa para evaluar la
calidad de cada división del árbol durante el entrenamiento.
Desde la versión interactiva: se selecciona al iniciar el programa.

MAX_PROFUNDIDAD (entero o None)
Valor: None (sin límite) o un número entero positivo (ej: 3)
Descripción: Limita cuántos niveles puede tener el árbol. Con None, el
árbol crece hasta que todos los nodos sean puros.
Desde la versión interactiva: se selecciona al iniciar el programa.

MIN_MUESTRAS_SPLIT (entero)
Valor por defecto: 2
Descripción: Número mínimo de muestras que debe tener un nodo para que
el algoritmo intente dividirlo. Aumentarlo simplifica el árbol.

MIN_MUESTRAS_HOJA (entero)
Valor por defecto: 1
Descripción: Número mínimo de muestras que debe tener una hoja (nodo
terminal). Aumentarlo puede reducir el overfitting.

RANDOM_STATE (entero)
Valor por defecto: 42
Descripción: Semilla de aleatoriedad. Garantiza que los resultados sean
reproducibles. Puede cambiarse a cualquier número entero.

HACER_SPLIT (booleano)
Valor por defecto: False
Descripción: Si se establece en True (y el dataset tiene más de 10 filas),
el sistema dividirá los datos en conjunto de entrenamiento y conjunto de
prueba (train/test split) para una evaluación más realista del modelo.

PROPORCION_TEST (decimal)
Valor por defecto: 0.2
Descripción: Proporción de datos reservada para el conjunto de prueba.
0.2 significa que el 20% de los datos se usará para evaluar el modelo y
el 80% para entrenarlo. Solo aplica si HACER_SPLIT = True.

================================================================================ 10. INTERPRETACIÓN DE LOS RESULTADOS
================================================================================

IMPORTANCIA DE ATRIBUTOS
Es un valor entre 0 y 1 que indica cuánto contribuye cada variable a las
decisiones del árbol durante el entrenamiento. La suma de todas las
importancias siempre es igual a 1. - Valor = 0 → El árbol nunca usó ese atributo para decidir. - Valor = 0.25 → El atributo contribuye un 25% a las decisiones. - Valor cercano a 1 → Es el atributo más importante del modelo.

PROBABILIDADES DE PREDICCIÓN (predict_proba)
Indican la distribución de clases en la hoja del árbol donde terminó el
caso evaluado. - P(clase) = 1.0 → Todos los datos históricos en esa hoja pertenecen
a esa clase. Predicción absoluta (árbol sin poda). - P(clase) = 0.75 → El 75% de los datos históricos en esa hoja
pertenecen a esa clase. Existe un 25% de incertidumbre. - P(clase) = 0.5 → Mezcla perfecta. El modelo no puede distinguir.

ACCURACY (Exactitud)
Porcentaje de predicciones correctas sobre el total de casos evaluados. - 1.0000 = 100% de aciertos. - 0.8500 = 85% de aciertos, 15% de errores.

PRECISION vs RECALL

- Precisión alta → El modelo no "da alarmas falsas". Cuando dice "Si",
  casi siempre es correcto.
- Recall alto → El modelo no "se pierde casos". Detecta la mayoría de
  los casos positivos reales.
- En diagnóstico médico o detección de fraude, el Recall suele ser más
  importante (es peor no detectar un caso real que dar una falsa alarma).

================================================================================ 11. PREGUNTAS FRECUENTES (FAQ)
================================================================================

P: ¿El programa no inicia y sale un error sobre módulos no encontrados?
R: Instale las dependencias necesarias con:
pip install pandas scikit-learn matplotlib
Si tiene múltiples versiones de Python, intente con:
pip3 install pandas scikit-learn matplotlib

P: ¿El programa dice "FileNotFoundError: datos_csv.csv"?
R: Asegúrese de que el archivo datos_csv.csv esté en la MISMA CARPETA que
arbolDecision.py. El programa debe ejecutarse desde esa carpeta.

P: ¿La ventana del árbol gráfico no se abre o cierra inmediatamente?
R: Esto puede ocurrir en algunos entornos de servidor o terminales sin
interfaz gráfica. La imagen del árbol siempre se guarda como
arbol_decision.png en la carpeta del proyecto, independientemente de
si la ventana se abre o no.

P: ¿Por qué el accuracy es 100% siempre?
R: Con un dataset pequeño (como el de 8 filas) y sin límite de profundidad,
el árbol puede memorizar todos los datos de entrenamiento, logrando un
100% de accuracy en ese mismo conjunto. Para una evaluación más realista,
active HACER_SPLIT = True y use un dataset más grande (más de 50 filas).

P: ¿Cómo puedo usar datos numéricos en lugar de texto (Si/No, Alta/Baja)?
R: El sistema admite valores numéricos directamente. No es necesario cambiar
nada especial; simplemente coloque los números en el dataset y en el
nuevo_caso. El sistema los procesará automáticamente.

P: ¿Puedo predecir más de dos clases (clasificación multiclase)?
R: Sí. El sistema soporta cualquier número de clases. Solo asegúrese de que
la columna target en su dataset tenga todos los valores posibles y que el
nuevo_caso también sea consistente con el dataset.

P: ¿Cómo cambio el caso que el sistema predice por defecto?
R: Modifique el diccionario "nuevo_caso" en la Sección 3 del código
(arbolDecision.py, aproximadamente líneas 194-199). También puede usar
el Simulador Interactivo sin modificar el código.

================================================================================ 12. GLOSARIO DE TÉRMINOS
================================================================================

Árbol de Decisión
Modelo de machine learning supervisado que divide recursivamente los datos
en subconjuntos mediante condiciones sobre atributos, hasta llegar a una
predicción final.

Atributo / Feature / Variable
Característica de entrada que el modelo usa para tomar decisiones
(ej: Fiebre, Tos, Dolor).

Clase / Target / Etiqueta
Variable de salida que el modelo intenta predecir
(ej: Enfermedad = Si o No).

Criterio de División
Métrica matemática que el árbol usa para elegir qué atributo y con qué
valor dividir los datos en cada nodo (Gini o Entropía).

DSS (Decision Support System)
Sistema de Soporte a la Decisión. Herramienta computacional que apoya
al directivo en la toma de decisiones basadas en datos.

Entropía
Métrica de la Teoría de la Información que mide el nivel de "desorden" o
incertidumbre en un conjunto de datos. El árbol busca minimizarla.

Gini (Índice de Impureza de Gini)
Métrica probabilística que mide la probabilidad de clasificar
incorrectamente un elemento aleatorio del nodo. El árbol busca minimizarlo.

Hoja (nodo hoja)
Nodo terminal del árbol que no tiene hijos. Representa la predicción final.

Importancia de Atributos
Medida de cuánto contribuye cada variable a las decisiones del árbol.
Un valor de 0 significa que la variable nunca fue usada.

Nodo Interno
División del árbol que evalúa una condición y dirige los datos hacia
dos o más ramas.

Nodo Raíz
El primer nodo del árbol; evalúa el atributo más importante de todo el
dataset.

Overfitting (Sobreajuste)
Cuando el modelo "memoriza" demasiado los datos de entrenamiento y pierde
capacidad de generalizar a datos nuevos.

Profundidad del Árbol
Número de niveles desde el nodo raíz hasta la hoja más profunda.

Simulador What-If
Herramienta interactiva del sistema que permite cambiar los valores de un
caso y observar cómo cambia la predicción del árbol.

Train/Test Split
Técnica de evaluación que divide el dataset en un conjunto de entrenamiento
(para entrenar el modelo) y un conjunto de prueba (para evaluar su
rendimiento con datos no vistos).

================================================================================
FIN DEL MANUAL DE USUARIO
Sistema de Soporte a la Decisión - Árbol de Decisión
================================================================================
