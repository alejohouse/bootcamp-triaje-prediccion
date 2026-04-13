"""
Modelo 2: Predicción de Nivel de Triaje.
Entrena y evalúa clasificadores para predecir el nivel de triaje
basado en diagnóstico sindrómico, sexo, edad y unidad de atención.
"""
import numpy as np
import pandas as pd
import os
import sys
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE, TRIAGE_LABELS


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Entrena múltiples modelos de triaje y compara rendimiento."""
    print("=" * 60)
    print("🚨 ENTRENAMIENTO: MODELO DE PREDICCIÓN DE TRIAJE")
    print("=" * 60)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'─' * 50}")
        print(f"🔄 Entrenando: {name}...")
        start = time.time()

        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average='weighted')
        f1_m = f1_score(y_test, y_pred, average='macro')

        results[name] = {
            'model': model,
            'accuracy': acc,
            'f1_weighted': f1_w,
            'f1_macro': f1_m,
            'train_time': train_time,
            'y_pred': y_pred,
        }

        print(f"   ⏱️ Tiempo: {train_time:.1f}s")
        print(f"   📊 Accuracy: {acc:.4f}")
        print(f"   📊 F1 (weighted): {f1_w:.4f}")
        print(f"   📊 F1 (macro): {f1_m:.4f}")

    return results


def select_best_model(results):
    """Selecciona el mejor modelo basado en F1 weighted."""
    best_name = max(results, key=lambda k: results[k]['f1_weighted'])
    best = results[best_name]

    print(f"\n{'=' * 60}")
    print(f"🏆 MEJOR MODELO DE TRIAJE: {best_name}")
    print(f"   Accuracy: {best['accuracy']:.4f}")
    print(f"   F1 (weighted): {best['f1_weighted']:.4f}")
    print(f"   F1 (macro): {best['f1_macro']:.4f}")
    print(f"{'=' * 60}")

    return best_name, best['model']


def save_model(model, model_name):
    """Guarda el modelo y metadatos."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, 'triage_model.pkl')
    joblib.dump(model, path)
    print(f"\n💾 Modelo guardado: {path}")

    with open(os.path.join(MODELS_DIR, 'triage_model_name.txt'), 'w') as f:
        f.write(model_name)


def plot_comparison(results):
    """Gráfico comparativo de modelos de triaje."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = list(results.keys())
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    # Accuracy
    accs = [results[n]['accuracy'] for n in names]
    bars = axes[0].bar(names, accs, color=colors, edgecolor='white')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy por Modelo', fontweight='bold')
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                    ha='center', fontweight='bold')

    # F1 Scores
    f1_w = [results[n]['f1_weighted'] for n in names]
    f1_m = [results[n]['f1_macro'] for n in names]
    x = np.arange(len(names))
    width = 0.35
    axes[1].bar(x - width/2, f1_w, width, label='F1 Weighted', color='#3498db', edgecolor='white')
    axes[1].bar(x + width/2, f1_m, width, label='F1 Macro', color='#e74c3c', edgecolor='white')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 Scores por Modelo', fontweight='bold')
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    # Tiempo
    times = [results[n]['train_time'] for n in names]
    bars3 = axes[2].bar(names, times, color=colors, edgecolor='white')
    axes[2].set_ylabel('Tiempo (s)')
    axes[2].set_title('Tiempo de Entrenamiento', fontweight='bold')
    for bar, val in zip(bars3, times):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{val:.1f}s',
                    ha='center', fontweight='bold')

    plt.tight_layout()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUTS_DIR, 'triage_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Comparación guardada en {OUTPUTS_DIR}/triage_model_comparison.png")


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Genera la matriz de confusión del mejor modelo."""
    labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Matriz de confusión (valores absolutos)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
               xticklabels=[f'T{l}' for l in labels],
               yticklabels=[f'T{l}' for l in labels])
    axes[0].set_xlabel('Predicho')
    axes[0].set_ylabel('Real')
    axes[0].set_title(f'Matriz de Confusión - {model_name}\n(Valores Absolutos)', fontweight='bold')

    # Matriz normalizada
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1],
               xticklabels=[f'T{l}' for l in labels],
               yticklabels=[f'T{l}' for l in labels])
    axes[1].set_xlabel('Predicho')
    axes[1].set_ylabel('Real')
    axes[1].set_title(f'Matriz de Confusión Normalizada\n(Proporción)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'triage_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Matriz de confusión guardada en {OUTPUTS_DIR}/triage_confusion_matrix.png")


if __name__ == "__main__":
    # Cargar datos preparados
    print("📂 Cargando datos preparados...")
    data = np.load(os.path.join(OUTPUTS_DIR, 'triage_data.npz'))
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"   Distribución triaje (test):")
    unique, counts = np.unique(y_test, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"     Triaje {u}: {c:,}")

    # Entrenar y evaluar
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Seleccionar mejor modelo
    best_name, best_model = select_best_model(results)

    # Guardar
    save_model(best_model, best_name)
    plot_comparison(results)
    plot_confusion_matrix(y_test, results[best_name]['y_pred'], best_name)

    # Classification report
    y_pred = results[best_name]['y_pred']
    print(f"\n📋 Classification Report ({best_name}):")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(OUTPUTS_DIR, 'triage_classification_report.csv'))
    print(report_df.to_string())
    print(f"\n💾 Reporte guardado en {OUTPUTS_DIR}/triage_classification_report.csv")

    print(f"\n{'=' * 60}")
    print(f"✅ MODELO DE TRIAJE ENTRENADO EXITOSAMENTE")
    print(f"{'=' * 60}")
