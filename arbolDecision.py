"""
================================================================================
ARBOL DE DECISION
================================================================================
Para adaptar este script a cualquier problema de clasificacion, solo debes
modificar las tres secciones marcadas con:

    >>> MODIFICAR AQUI <<<

El resto del script funciona automaticamente.
================================================================================
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIGURACION GENERAL DEL MODELO
# Puedes ajustar estos hiperparametros sin tocar nada mas.
# ==============================================================================

# B1 - El usuario elige el Criterio de División (Gini vs. Entropía)

while True:
    criterio_input = input("Seleccione el criterio de division (gini / entropy): ").strip().lower()
    
    if criterio_input in ["gini", "entropy"]:
        CRITERIO = criterio_input
        break
    else:
        print("Error: Debe escribir 'gini' o 'entropy'. Intente nuevamente.\n")

# B2 - Selección de profundidad del arbol

while True:
    profundidad_input = input("Ingrese la profundidad maxima del arbol (Enter para sin limite): ").strip()
    
    if profundidad_input == "":
        MAX_PROFUNDIDAD = None
        break
    
    if profundidad_input.isdigit():
        profundidad_val = int(profundidad_input)
        if profundidad_val > 0:
            MAX_PROFUNDIDAD = profundidad_val
            break
    
    print("Error: Ingrese un numero entero positivo o deje vacio para sin limite.\n")
# ------------------------------------------------------------------------------

MIN_MUESTRAS_SPLIT = 2      # minimo de muestras para dividir un nodo
MIN_MUESTRAS_HOJA  = 1      # minimo de muestras en una hoja
RANDOM_STATE    = 42
HACER_SPLIT     = False     # True si tienes suficientes datos (recomendado >50 filas)
PROPORCION_TEST = 0.2       # solo se usa si HACER_SPLIT = True


# ==============================================================================
# >>> MODIFICAR AQUI <<< SECCION 1: DATASET
#
# Opcion A: cargar desde archivo (poner el csv afuera de la carpeta de arbolDecision.py)
df = pd.read_csv("datos_csv.csv")
data = df.to_dict(orient="list")


#
# Opcion B: definir los datos directamente (como en este ejemplo)
#
# Reglas:
#   - Cada clave del diccionario es una columna.
#   - Las columnas pueden ser categoricas (strings) o numericas.
#   - La columna target (lo que quieres predecir) puede llamarse como quieras,
#     solo asegurate de escribir el mismo nombre en SECCION 2.
# ==============================================================================

data = {
    "Fiebre":       ["Alta", "Alta", "Alta", "Alta", "Baja", "Baja", "Baja", "Baja"],
    "Tos":          ["Si",   "Si",   "No",   "No",   "Si",   "Si",   "No",   "No"],
    "Dolor":        ["Si",   "No",   "Si",   "No",   "Si",   "No",   "Si",   "No"],
    "EdadMayor50":  ["Si",   "No",   "Si",   "No",   "No",   "Si",   "No",   "Si"],
    "Enfermedad":   ["Si",   "Si",   "Si",   "No",   "No",   "No",   "No",   "No"],
}

#df = pd.DataFrame(data)


# ==============================================================================
# >>> MODIFICAR AQUI <<< SECCION 2: NOMBRE DE LA COLUMNA TARGET
#
# Escribe exactamente el nombre de la columna que quieres predecir.
# Debe existir en el dataset de arriba.
# ==============================================================================

COLUMNA_TARGET = "Enfermedad"


# ==============================================================================
# >>> MODIFICAR AQUI <<< SECCION 3: NUEVO CASO A PREDECIR
#
# Escribe los valores del caso que quieres clasificar.
# Las claves deben ser exactamente las mismas columnas del dataset
# (excepto la columna target, que es lo que el modelo va a predecir).
# ==============================================================================

nuevo_caso = {
    "Fiebre":       ["Alta"],
    "Tos":          ["Si"],
    "Dolor":        ["No"],
    "EdadMayor50":  ["No"],
}


# ==============================================================================
# A PARTIR DE AQUI NO ES NECESARIO MODIFICAR NADA
# ==============================================================================


def preparar_datos(df, columna_target):
    """
    Separa features de target y codifica variables categoricas.
    Retorna X codificado, y numerico, y los nombres de las clases.
    """
    X_raw = df.drop(columna_target, axis=1)
    y_raw = df[columna_target]

    # Codificar features categoricas
    columnas_categoricas = X_raw.select_dtypes(include="object").columns
    columnas_numericas   = X_raw.select_dtypes(exclude="object").columns

    X_cat = pd.get_dummies(X_raw[columnas_categoricas], drop_first=True) if len(columnas_categoricas) > 0 else pd.DataFrame()
    X_num = X_raw[columnas_numericas].reset_index(drop=True)
    X = pd.concat([X_num, X_cat], axis=1)

    # Codificar target
    if y_raw.dtype == object:
        clases      = sorted(y_raw.unique())
        clase_ref   = clases[0]
        y = (y_raw != clase_ref).astype(int)
    else:
        clases    = sorted(y_raw.unique())
        clase_ref = clases[0]
        y = (y_raw != clase_ref).astype(int)

    nombres_clases = [str(c) for c in clases]
    return X, y, nombres_clases


def codificar_nuevo_caso(nuevo_caso_dict, X_entrenamiento):
    """
    Codifica un nuevo caso con el mismo esquema que los datos de entrenamiento.
    Usa reindex para garantizar que las columnas coincidan exactamente.
    """
    nuevo_df  = pd.DataFrame(nuevo_caso_dict)
    nuevo_enc = pd.get_dummies(nuevo_df, drop_first=True)
    nuevo_enc = nuevo_enc.reindex(columns=X_entrenamiento.columns, fill_value=0)
    return nuevo_enc


def entrenar_modelo(X_train, y_train):
    modelo = DecisionTreeClassifier(
        criterion=CRITERIO,
        max_depth=MAX_PROFUNDIDAD,
        min_samples_split=MIN_MUESTRAS_SPLIT,
        min_samples_leaf=MIN_MUESTRAS_HOJA,
        random_state=RANDOM_STATE
    )
    modelo.fit(X_train, y_train)
    return modelo


def mostrar_dataset(df):
    sep = "=" * 55
    print(sep)
    print("DATASET")
    print(sep)
    print(df.to_string(index=True))
    print(f"\nTotal filas: {len(df)}")
    print()


def mostrar_features(X, y, nombres_clases):
    sep = "=" * 55
    print(sep)
    print("FEATURES CODIFICADAS")
    print(sep)
    print(X.to_string(index=True))
    print(f"\nColumnas: {list(X.columns)}")
    print(f"Target:   {list(y)}  (clases: {nombres_clases})")
    print()


def mostrar_modelo(modelo, X):
    sep = "=" * 55
    print(sep)
    print("MODELO ENTRENADO")
    print(sep)
    print(f"  Profundidad:  {modelo.get_depth()}")
    print(f"  Num. hojas:   {modelo.get_n_leaves()}")
    print(f"  Criterio:     {CRITERIO}")
    print(f"  Max depth:    {MAX_PROFUNDIDAD}")
    print()
    print("IMPORTANCIA DE ATRIBUTOS")
    print(sep)
    importancias = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
    for nombre, valor in importancias.items():
        barra = "#" * int(valor * 40)
        print(f"  {nombre:<30} {valor:.4f}  {barra}")
    print()


def visualizar_arbol(modelo, X, nombres_clases):
    plt.figure(figsize=(16, 8))
    plot_tree(
        modelo,
        feature_names=X.columns,
        class_names=nombres_clases,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title(f"Arbol de Decision  |  criterio={CRITERIO}  |  max_depth={MAX_PROFUNDIDAD}", fontsize=13)
    plt.tight_layout()
    plt.savefig("arbol_decision.png", dpi=150)
    plt.show()
    print("Arbol guardado como: arbol_decision.png\n")


def predecir(modelo, nuevo_caso_dict, X_entrenamiento, nombres_clases):
    sep = "=" * 55
    print(sep)
    print("PREDICCION - NUEVO CASO")
    print(sep)
    print("Valores de entrada:")
    for k, v in nuevo_caso_dict.items():
        print(f"  {k}: {v[0]}")
    print()

    nuevo_enc  = codificar_nuevo_caso(nuevo_caso_dict, X_entrenamiento)
    prediccion = modelo.predict(nuevo_enc)[0]
    probabilidades = modelo.predict_proba(nuevo_enc)[0]

    clase_predicha = nombres_clases[prediccion]
    print(f"  Clase predicha: {clase_predicha}")
    print()
    for i, nombre in enumerate(nombres_clases):
        print(f"  P({nombre}): {probabilidades[i]:.4f}")
    print()


def evaluar(modelo, X, y, X_test=None, y_test=None, nombres_clases=None):
    sep = "=" * 55
    print(sep)
    datos_eval = (X_test, y_test) if X_test is not None else (X, y)
    etiqueta   = "TEST" if X_test is not None else "ENTRENAMIENTO (referencia)"
    print(f"EVALUACION ({etiqueta})")
    print(sep)

    y_pred = modelo.predict(datos_eval[0])
    print(f"  Accuracy: {accuracy_score(datos_eval[1], y_pred):.4f}\n")
    print(classification_report(datos_eval[1], y_pred, target_names=nombres_clases))


def mostrar_reglas(modelo, X):
    sep = "=" * 55
    print(sep)
    print("REGLAS DEL ARBOL (texto)")
    print(sep)
    reglas = export_text(modelo, feature_names=list(X.columns))
    print(reglas)


# ==============================================================================
# EJECUCION PRINCIPAL
# ==============================================================================

mostrar_dataset(df)

X, y, nombres_clases = preparar_datos(df, COLUMNA_TARGET)
mostrar_features(X, y, nombres_clases)

if HACER_SPLIT and len(df) > 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PROPORCION_TEST, random_state=RANDOM_STATE
    )
    modelo = entrenar_modelo(X_train, y_train)
    mostrar_modelo(modelo, X)
    visualizar_arbol(modelo, X, nombres_clases)
    predecir(modelo, nuevo_caso, X, nombres_clases)
    evaluar(modelo, X, y, X_test, y_test, nombres_clases)
else:
    modelo = entrenar_modelo(X, y)
    mostrar_modelo(modelo, X)
    visualizar_arbol(modelo, X, nombres_clases)
    predecir(modelo, nuevo_caso, X, nombres_clases)
    evaluar(modelo, X, y, nombres_clases=nombres_clases)

mostrar_reglas(modelo, X)