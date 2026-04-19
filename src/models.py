"""
Entrenamiento y evaluación de modelos predictivos de enfermedad y triaje.

Uso:
    python src/models.py --tipo enfermedad
    python src/models.py --tipo triaje
"""
import argparse
import os
import sys
import time
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import LR_PARAMS, MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE, RF_PARAMS


# ============================================================
# NÚCLEO GENÉRICO — compartido por enfermedad y triaje
# ============================================================

def entrenar_y_comparar(modelos_config, y_train, y_test):
    """
    Entrena y evalúa un conjunto de modelos de forma genérica.

    Args:
        modelos_config: dict con formato {nombre: (modelo, X_train, X_test)}.
                        Cada modelo recibe sus propios datos (útil cuando
                        algunos modelos requieren escalado y otros no).
        y_train: etiquetas de entrenamiento.
        y_test:  etiquetas de prueba.

    Returns:
        dict con métricas y artefactos por modelo.
    """
    resultados = {}

    for nombre, (modelo, X_tr, X_te) in modelos_config.items():
        print(f"\n{'─' * 50}")
        print(f"🔄 Entrenando: {nombre}...")
        inicio = time.time()

        modelo.fit(X_tr, y_train)
        tiempo = time.time() - inicio

        y_pred = modelo.predict(X_te)
        acc    = accuracy_score(y_test, y_pred)
        f1_w   = f1_score(y_test, y_pred, average='weighted')
        f1_m   = f1_score(y_test, y_pred, average='macro')

        top3_acc = None
        if hasattr(modelo, 'predict_proba'):
            probas   = modelo.predict_proba(X_te)
            top3_acc = top_k_accuracy_score(y_test, probas, k=3)

        resultados[nombre] = {
            'modelo':       modelo,
            'X_test':       X_te,
            'accuracy':     acc,
            'f1_ponderado': f1_w,
            'f1_macro':     f1_m,
            'top3_accuracy': top3_acc,
            'tiempo':       tiempo,
            'y_pred':       y_pred,
        }

        print(f"   ⏱️  Tiempo:       {tiempo:.1f}s")
        print(f"   📊 Accuracy:     {acc:.4f}")
        print(f"   📊 F1 ponderado: {f1_w:.4f}")
        print(f"   📊 F1 macro:     {f1_m:.4f}")
        if top3_acc:
            print(f"   📊 Top-3 Acc:    {top3_acc:.4f}")

    return resultados


def seleccionar_mejor_modelo(resultados):
    """Selecciona el modelo con mayor F1 ponderado."""
    mejor_nombre = max(resultados, key=lambda k: resultados[k]['f1_ponderado'])
    mejor        = resultados[mejor_nombre]

    print(f"\n{'=' * 60}")
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   Accuracy:     {mejor['accuracy']:.4f}")
    print(f"   F1 ponderado: {mejor['f1_ponderado']:.4f}")
    if mejor['top3_accuracy']:
        print(f"   Top-3 Acc:    {mejor['top3_accuracy']:.4f}")
    print(f"{'=' * 60}")

    return mejor_nombre, mejor['modelo']


def guardar_modelo(modelo, nombre_modelo, prefijo, scaler=None):
    """
    Guarda el modelo entrenado con prefijo de dominio ('enfermedad' o 'triaje').

    Args:
        modelo:        objeto entrenado.
        nombre_modelo: nombre del algoritmo ganador (para el .txt de metadatos).
        prefijo:       'enfermedad' o 'triaje'.
        scaler:        scaler opcional (solo si el modelo requiere escalado).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    ruta = os.path.join(MODELS_DIR, f'{prefijo}_model.pkl')
    joblib.dump(modelo, ruta)
    print(f"\n💾 Modelo guardado: {ruta}")

    with open(os.path.join(MODELS_DIR, f'{prefijo}_model_name.txt'), 'w') as f:
        f.write(nombre_modelo)

    if scaler is not None:
        ruta_scaler = os.path.join(MODELS_DIR, f'{prefijo}_scaler.pkl')
        joblib.dump(scaler, ruta_scaler)
        print(f"💾 Scaler guardado: {ruta_scaler}")


def plot_comparison(resultados, output_path):
    """Gráfico comparativo de Accuracy, F1 ponderado y Tiempo (genérico)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    nombres = list(resultados.keys())
    colores = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    # --- Accuracy ---
    accs = [resultados[n]['accuracy'] for n in nombres]
    bars = axes[0].bar(nombres, accs, color=colores[:len(nombres)], edgecolor='white')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy por Modelo', fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                     f'{val:.3f}', ha='center', fontweight='bold')

    # --- F1 ponderado ---
    f1s  = [resultados[n]['f1_ponderado'] for n in nombres]
    bars2 = axes[1].bar(nombres, f1s, color=colores[:len(nombres)], edgecolor='white')
    axes[1].set_ylabel('F1 Ponderado')
    axes[1].set_title('F1 Score por Modelo', fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars2, f1s):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                     f'{val:.3f}', ha='center', fontweight='bold')

    # --- Tiempo ---
    tiempos = [resultados[n]['tiempo'] for n in nombres]
    bars3   = axes[2].bar(nombres, tiempos, color=colores[:len(nombres)], edgecolor='white')
    axes[2].set_ylabel('Tiempo (s)')
    axes[2].set_title('Tiempo de Entrenamiento', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars3, tiempos):
        axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                     f'{val:.1f}s', ha='center', fontweight='bold')

    plt.tight_layout()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Comparación guardada en {output_path}")


def exportar_reporte(y_test, y_pred, clases, output_path):
    """Exporta el reporte de clasificación como CSV."""
    reporte = classification_report(
        y_test, y_pred, target_names=clases, output_dict=True
    )
    pd.DataFrame(reporte).transpose().to_csv(output_path)
    print(f"   📄 Reporte guardado en {output_path}")


# ============================================================
# ENTRENAMIENTO DE ENFERMEDAD (específico)
# ============================================================

def _run_enfermedad():
    """Pipeline completo de entrenamiento del modelo de enfermedad."""
    from src.data_preparation import load_enfermedad_dataset, prepare_enfermedad_data

    print("=" * 60)
    print("🏥 ENTRENAMIENTO: MODELO DE PREDICCIÓN DE ENFERMEDAD")
    print("=" * 60)

    df_enfermedades, sintomas_cols, _ = load_enfermedad_dataset()
    X_train, X_test, y_train, y_test, le_enfermedad = prepare_enfermedad_data(
        df_enfermedades, sintomas_cols
    )

    # Guardar columnas de síntomas para el pipeline
    joblib.dump(sintomas_cols, os.path.join(MODELS_DIR, 'sintomas_columns.pkl'))

    # Scaler solo para Logistic Regression (los árboles no lo necesitan)
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    modelos_config = {
        "Random Forest": (
            RandomForestClassifier(**RF_PARAMS),
            X_train, X_test,
        ),
        "Extra Trees": (
            ExtraTreesClassifier(
                n_estimators=200, max_depth=20, min_samples_split=5,
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            X_train, X_test,
        ),
        "Logistic Regression": (
            LogisticRegression(**LR_PARAMS),
            X_train_sc, X_test_sc,
        ),
    }

    resultados   = entrenar_y_comparar(modelos_config, y_train, y_test)
    mejor_nombre, mejor_modelo = seleccionar_mejor_modelo(resultados)

    # Guardar modelo (y scaler solo si ganó Logistic Regression)
    mejor_scaler = scaler if mejor_nombre == "Logistic Regression" else None
    guardar_modelo(mejor_modelo, mejor_nombre, prefijo='enfermedad', scaler=mejor_scaler)

    # Outputs de script standalone
    plot_comparison(
        resultados,
        os.path.join(OUTPUTS_DIR, 'enfermedad_model_comparison.png'),
    )
    _plot_importancia_features(mejor_modelo, sintomas_cols)
    exportar_reporte(
        y_test,
        resultados[mejor_nombre]['y_pred'],
        le_enfermedad.classes_,
        os.path.join(OUTPUTS_DIR, 'enfermedad_classification_report.csv'),
    )

    mejor = resultados[mejor_nombre]
    print(f"\n{'=' * 60}")
    print(f"✅ MODELO DE ENFERMEDAD ENTRENADO")
    print(f"   Modelo:       {mejor_nombre}")
    print(f"   Accuracy:     {mejor['accuracy']:.4f}")
    print(f"   F1 ponderado: {mejor['f1_ponderado']:.4f}")
    print(f"{'=' * 60}")


def _plot_importancia_features(modelo, sintomas_cols, top_n=30):
    """Visualiza los síntomas más importantes (solo para modelos de árboles)."""
    if not hasattr(modelo, 'feature_importances_'):
        return

    importancias = modelo.feature_importances_
    indices      = np.argsort(importancias)[-top_n:]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(top_n), importancias[indices],
            color=plt.cm.viridis(np.linspace(0.3, 0.9, top_n)), edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([sintomas_cols[i] for i in indices], fontsize=9)
    ax.set_xlabel('Importancia')
    ax.set_title(f'Top {top_n} Síntomas Más Importantes', fontweight='bold')

    plt.tight_layout()
    ruta = os.path.join(OUTPUTS_DIR, 'enfermedad_feature_importance.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Feature importance guardado en {ruta}")


# ============================================================
# ENTRENAMIENTO DE TRIAJE (específico)
# ============================================================

def _run_triaje():
    """Pipeline completo de entrenamiento del modelo de triaje."""
    from src.data_preparation import (
        encode_morbilidad_features,
        load_morbilidad_dataset,
        prepare_triaje_data,
    )

    print("=" * 60)
    print("🚨 ENTRENAMIENTO: MODELO DE PREDICCIÓN DE TRIAJE")
    print("=" * 60)

    df_morb             = load_morbilidad_dataset()
    df_morb_encoded, _  = encode_morbilidad_features(df_morb)
    X_train, X_test, y_train, y_test, _ = prepare_triaje_data(df_morb_encoded)

    # Los datos ya vienen escalados desde prepare_triaje_data;
    # todos los modelos usan el mismo X_train / X_test.
    modelos_config = {
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1,
            ),
            X_train, X_test,
        ),
        "Extra Trees": (
            ExtraTreesClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1,
            ),
            X_train, X_test,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.1,
                random_state=RANDOM_STATE,
            ),
            X_train, X_test,
        ),
        "Logistic Regression": (
            LogisticRegression(
                **LR_PARAMS,
                class_weight='balanced',
            ),
            X_train, X_test,
        ),
    }

    resultados   = entrenar_y_comparar(modelos_config, y_train, y_test)
    mejor_nombre, mejor_modelo = seleccionar_mejor_modelo(resultados)

    guardar_modelo(mejor_modelo, mejor_nombre, prefijo='triaje')

    # Outputs de script standalone
    plot_comparison(
        resultados,
        os.path.join(OUTPUTS_DIR, 'triaje_model_comparison.png'),
    )
    _plot_confusion_matrix(y_test, resultados[mejor_nombre]['y_pred'], mejor_nombre)
    exportar_reporte(
        y_test,
        resultados[mejor_nombre]['y_pred'],
        [str(c) for c in sorted(np.unique(y_test))],
        os.path.join(OUTPUTS_DIR, 'triaje_classification_report.csv'),
    )

    mejor = resultados[mejor_nombre]
    print(f"\n{'=' * 60}")
    print(f"✅ MODELO DE TRIAJE ENTRENADO")
    print(f"   Modelo:       {mejor_nombre}")
    print(f"   Accuracy:     {mejor['accuracy']:.4f}")
    print(f"   F1 ponderado: {mejor['f1_ponderado']:.4f}")
    print(f"{'=' * 60}")


def _plot_confusion_matrix(y_test, y_pred, nombre_modelo):
    """Genera la matriz de confusión del mejor modelo de triaje."""
    etiquetas = sorted(np.unique(np.concatenate([y_test, y_pred])))
    cm        = confusion_matrix(y_test, y_pred, labels=etiquetas)
    cm_norm   = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=[f'T{l}' for l in etiquetas],
                yticklabels=[f'T{l}' for l in etiquetas])
    axes[0].set_xlabel('Predicho')
    axes[0].set_ylabel('Real')
    axes[0].set_title(f'Matriz de Confusión — {nombre_modelo}', fontweight='bold')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1],
                xticklabels=[f'T{l}' for l in etiquetas],
                yticklabels=[f'T{l}' for l in etiquetas])
    axes[1].set_xlabel('Predicho')
    axes[1].set_ylabel('Real')
    axes[1].set_title('Matriz Normalizada', fontweight='bold')

    plt.tight_layout()
    ruta = os.path.join(OUTPUTS_DIR, 'triaje_confusion_matrix.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Matriz de confusión guardada en {ruta}")


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento del modelo predictivo de enfermedad o triaje"
    )
    parser.add_argument(
        "--tipo",
        choices=["enfermedad", "triaje"],
        required=True,
        help="Tipo de modelo a entrenar: 'enfermedad' o 'triaje'",
    )
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if args.tipo == "enfermedad":
        _run_enfermedad()
    else:
        _run_triaje()
