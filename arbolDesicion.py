# =============================================================
# Árbol de Decisión para Aprobación de Crédito
# Basado en: Turban — Cap. 4 / GeeksForGeeks Decision Tree
# Modelo: Clasificación supervisada con sklearn
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# ── PASO 1: Crear dataset de empresa ──────────────────────────
def crear_dataset():
    """
    Datos históricos de solicitudes de crédito.
    Variables:
      - ingresos: Alto / Medio / Bajo
      - historial: Bueno / Malo
      - empleo: Estable / Inestable
    Etiqueta:
      - aprobado: Si / No
    """
    datos = {
        'ingresos':   ['Alto','Bajo','Medio','Alto','Bajo','Alto','Medio','Medio',
                      'Alto','Medio','Bajo','Alto','Medio','Bajo','Medio','Alto'],
        'historial':  ['Bueno','Malo','Bueno','Malo','Bueno','Bueno','Malo','Bueno',
                      'Bueno','Malo','Bueno','Malo','Bueno','Malo','Bueno','Bueno'],
        'empleo':     ['Estable','Inestable','Estable','Estable','Inestable','Inestable','Estable','Inestable',
                      'Estable','Inestable','Estable','Inestable','Estable','Estable','Inestable','Estable'],
        'aprobado':   ['Si','No','Si','No','No','Si','No','Si',
                      'Si','No','Si','No','Si','No','Si','Si'],
    }
    df = pd.DataFrame(datos)
    print(" Dataset cargado:")
    print(df.to_string(index=False))
    print(f"\n Total: {len(df)} registros\n")
    return df


# ── PASO 2: Codificar variables de texto a números ────────────
def codificar_datos(df):
    """
    Los árboles de sklearn necesitan números, no texto.
    Convertimos: Alto→0, Bajo→1, Medio→2 etc.
    """
    encoders = {}
    df_encoded = df.copy()

    for col in df.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df_encoded, encoders


# ── PASO 3: Separar características y etiqueta ────────────────
def separar_datos(df_encoded):
    """
    X = variables predictoras (ingresos, historial, empleo)
    y = lo que queremos predecir (aprobado: Si/No)
    """
    X = df_encoded.drop(columns=['aprobado'])
    y = df_encoded['aprobado']

    # 70% entrenamiento, 30% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"  Entrenamiento: {len(X_train)} registros")
    print(f" Prueba: {len(X_test)} registros\n")
    return X_train, X_test, y_train, y_test


# ── PASO 4: Entrenar el árbol con criterio elegido ────────────
def entrenar_arbol(X_train, y_train, criterio='gini'):
    """
    criterio: 'gini' (Índice Gini) o 'entropy' (Ganancia Info)
    max_depth: profundidad máxima — evita overfitting
    min_samples_leaf: mínimo de muestras por hoja
    """
    modelo = DecisionTreeClassifier(
        criterion=criterio,
        max_depth=3,
        min_samples_leaf=2,
        random_state=42
    )
    modelo.fit(X_train, y_train)
    print(f" Árbol entrenado con criterio: {criterio.upper()}")
    return modelo


# ── PASO 5: Evaluar el modelo ─────────────────────────────────
def evaluar_modelo(modelo, X_test, y_test, criterio):
    """Muestra métricas de rendimiento del árbol."""
    y_pred = modelo.predict(X_test)

    print(f"\n{'='*45}")
    print(f"   EVALUACIÓN — {criterio.upper()}")
    print(f"{'='*45}")
    print(f"\n Precisión: {accuracy_score(y_test, y_pred) * 100:.1f}%")
    print("\n Reporte detallado:")
    print(classification_report(y_test, y_pred))
    print(" Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    return y_pred


# ── PASO 6: Predecir un cliente nuevo ─────────────────────────
def predecir_nuevo_cliente(modelo, encoders):
    """Ingresa manualmente un cliente y predice si aprueba."""
    print("\n PREDICCIÓN DE CLIENTE NUEVO")
    print("-" * 40)

    # Datos del cliente nuevo (puedes cambiarlos)
    cliente = {
        'ingresos':  'Alto',      # Alto / Medio / Bajo
        'historial': 'Bueno',     # Bueno / Malo
        'empleo':    'Estable',   # Estable / Inestable
    }

    print(f"Ingresos:  {cliente['ingresos']}")
    print(f"Historial: {cliente['historial']}")
    print(f"Empleo:    {cliente['empleo']}")

    # Codificar los valores del cliente nuevo
    cliente_codificado = [
        encoders['ingresos'].transform([cliente['ingresos']])[0],
        encoders['historial'].transform([cliente['historial']])[0],
        encoders['empleo'].transform([cliente['empleo']])[0],
    ]

    prediccion_cod = modelo.predict([cliente_codificado])[0]
    resultado = encoders['aprobado'].inverse_transform([prediccion_cod])[0]

    print(f"\n DECISIÓN DEL SISTEMA: {resultado}")
    if resultado == 'Si':
        print("    Crédito APROBADO")
    else:
        print("    Crédito RECHAZADO")


# ── PASO 7: Visualizar el árbol ───────────────────────────────
def visualizar_arbol(modelo, criterio):
    plt.figure(figsize=(14, 8))
    plot_tree(
        modelo,
        feature_names=['ingresos', 'historial', 'empleo'],
        class_names=['No', 'Si'],
        filled=True,
        rounded=True,
        fontsize=11
    )
    plt.title(f"Árbol de Decisión — Aprobación de Crédito ({criterio.upper()})",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"arbol_{criterio}.png", dpi=150)
    print(f"\n  Árbol guardado como: arbol_{criterio}.png")
    plt.show()


# ── PROGRAMA PRINCIPAL ────────────────────────────────────────
if __name__ == "__main__":

    print(" SISTEMA DE APROBACIÓN DE CRÉDITO")
    print("   Árbol de Decisión — Basado en Turban Cap. 4\n")

    # 1. Cargar y codificar datos
    df = crear_dataset()
    df_enc, encoders = codificar_datos(df)

    # 2. Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = separar_datos(df_enc)

    # 3. Entrenar con Gini y con Entropía
    arbol_gini     = entrenar_arbol(X_train, y_train, criterio='gini')
    arbol_entropia = entrenar_arbol(X_train, y_train, criterio='entropy')

    # 4. Evaluar ambos modelos
    evaluar_modelo(arbol_gini,     X_test, y_test, 'gini')
    evaluar_modelo(arbol_entropia, X_test, y_test, 'entropy')

    # 5. Predecir cliente nuevo
    predecir_nuevo_cliente(arbol_gini, encoders)

    # 6. Visualizar árboles
    visualizar_arbol(arbol_gini,     'gini')
    visualizar_arbol(arbol_entropia, 'entropy')