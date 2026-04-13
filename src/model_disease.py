"""
Modelo 1: Predicción de Enfermedad basado en síntomas.
Entrena y evalúa múltiples clasificadores para predecir la enfermedad
a partir de un vector binario de síntomas.
"""
import numpy as np
import pandas as pd
import os
import sys
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, top_k_accuracy_score
)
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE, CV_FOLDS, RF_PARAMS, XGB_PARAMS


def train_and_evaluate_models(X_train, X_test, y_train, y_test, le_disease):
    """Entrena múltiples modelos y compara su rendimiento."""
    print("=" * 60)
    print("🏥 ENTRENAMIENTO: MODELO DE PREDICCIÓN DE ENFERMEDAD")
    print("=" * 60)

    models = {
        "Random Forest": RandomForestClassifier(**RF_PARAMS),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'─' * 50}")
        print(f"🔄 Entrenando: {name}...")
        start = time.time()

        model.fit(X_train, y_train)
        train_time = time.time() - start

        # Predicciones
        y_pred = model.predict(X_test)

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average='weighted')
        f1_m = f1_score(y_test, y_pred, average='macro')

        # Top-3 accuracy (si el modelo soporta predict_proba)
        top3_acc = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            top3_acc = top_k_accuracy_score(y_test, y_proba, k=3)

        results[name] = {
            'model': model,
            'accuracy': acc,
            'f1_weighted': f1_w,
            'f1_macro': f1_m,
            'top3_accuracy': top3_acc,
            'train_time': train_time,
            'y_pred': y_pred
        }

        print(f"   ⏱️ Tiempo: {train_time:.1f}s")
        print(f"   📊 Accuracy: {acc:.4f}")
        print(f"   📊 F1 (weighted): {f1_w:.4f}")
        print(f"   📊 F1 (macro): {f1_m:.4f}")
        if top3_acc:
            print(f"   📊 Top-3 Accuracy: {top3_acc:.4f}")

    return results


def select_best_model(results):
    """Selecciona el mejor modelo basado en F1 weighted."""
    best_name = max(results, key=lambda k: results[k]['f1_weighted'])
    best = results[best_name]

    print(f"\n{'=' * 60}")
    print(f"🏆 MEJOR MODELO: {best_name}")
    print(f"   Accuracy: {best['accuracy']:.4f}")
    print(f"   F1 (weighted): {best['f1_weighted']:.4f}")
    if best['top3_accuracy']:
        print(f"   Top-3 Accuracy: {best['top3_accuracy']:.4f}")
    print(f"{'=' * 60}")

    return best_name, best['model']


def save_model(model, model_name):
    """Guarda el modelo entrenado."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, 'disease_model.pkl')
    joblib.dump(model, path)
    print(f"\n💾 Modelo guardado: {path}")

    # Guardar nombre del modelo
    with open(os.path.join(MODELS_DIR, 'disease_model_name.txt'), 'w') as f:
        f.write(model_name)


def plot_comparison(results):
    """Genera gráfico comparativo de modelos."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    names = list(results.keys())
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    # Accuracy
    accs = [results[n]['accuracy'] for n in names]
    bars1 = axes[0].bar(names, accs, color=colors[:len(names)], edgecolor='white')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy por Modelo', fontweight='bold')
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars1, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                    ha='center', fontweight='bold')

    # F1 Score
    f1s = [results[n]['f1_weighted'] for n in names]
    bars2 = axes[1].bar(names, f1s, color=colors[:len(names)], edgecolor='white')
    axes[1].set_ylabel('F1 Score (weighted)')
    axes[1].set_title('F1 Score por Modelo', fontweight='bold')
    axes[1].set_ylim(0, 1)
    for bar, val in zip(bars2, f1s):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                    ha='center', fontweight='bold')

    # Tiempo de entrenamiento
    times = [results[n]['train_time'] for n in names]
    bars3 = axes[2].bar(names, times, color=colors[:len(names)], edgecolor='white')
    axes[2].set_ylabel('Tiempo (segundos)')
    axes[2].set_title('Tiempo de Entrenamiento', fontweight='bold')
    for bar, val in zip(bars3, times):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}s',
                    ha='center', fontweight='bold')

    plt.tight_layout()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUTS_DIR, 'disease_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Gráfico comparativo guardado en {OUTPUTS_DIR}/disease_model_comparison.png")


def plot_top_features(model, symptom_cols, top_n=30):
    """Visualiza los síntomas más importantes para el modelo."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(top_n), importances[indices],
               color=plt.cm.viridis(np.linspace(0.3, 0.9, top_n)), edgecolor='white')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([symptom_cols[i] for i in indices], fontsize=9)
        ax.set_xlabel('Importancia')
        ax.set_title(f'Top {top_n} Síntomas Más Importantes para el Modelo', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, 'disease_feature_importance.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Feature importance guardado en {OUTPUTS_DIR}/disease_feature_importance.png")


if __name__ == "__main__":
    # Cargar datos preparados
    print("📂 Cargando datos preparados...")
    data = np.load(os.path.join(OUTPUTS_DIR, 'disease_data.npz'))
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    le_disease = joblib.load(os.path.join(MODELS_DIR, 'disease_label_encoder.pkl'))
    symptom_cols = joblib.load(os.path.join(MODELS_DIR, 'symptom_columns.pkl'))

    print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"   Clases: {len(le_disease.classes_)}")

    # Entrenar y evaluar
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, le_disease)

    # Seleccionar mejor modelo
    best_name, best_model = select_best_model(results)

    # Guardar
    save_model(best_model, best_name)
    plot_comparison(results)
    plot_top_features(best_model, symptom_cols)

    # Classification report del mejor modelo
    y_pred = results[best_name]['y_pred']
    print(f"\n📋 Classification Report ({best_name}):")
    report = classification_report(
        y_test, y_pred,
        target_names=le_disease.classes_,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(OUTPUTS_DIR, 'disease_classification_report.csv'))
    print(f"   💾 Reporte guardado en {OUTPUTS_DIR}/disease_classification_report.csv")

    # Mostrar resumen
    print(f"\n{'=' * 60}")
    print(f"✅ MODELO DE ENFERMEDAD ENTRENADO EXITOSAMENTE")
    print(f"   Modelo: {best_name}")
    print(f"   Accuracy: {results[best_name]['accuracy']:.4f}")
    print(f"   F1 (weighted): {results[best_name]['f1_weighted']:.4f}")
    print(f"{'=' * 60}")
