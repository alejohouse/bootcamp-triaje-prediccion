"""
Evaluación completa de los modelos.
Compara predicciones del pipeline con los diagnósticos reales del hospital.
"""
import numpy as np
import pandas as pd
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MODELS_DIR, OUTPUTS_DIR, MORBILIDAD_XLSX,
    TRIAJE_EXCLUIDO, SINDROMATICO_A_SINTOMAS
)
from src.pipeline import MedicalPredictionPipeline
from src.data_preparation import dx_to_sintoma_vector


def evaluar_enfermedad_en_datos_hospital(pipeline, df_urgencias, sample_size=500):
    """
    Evalúa el modelo de enfermedad contra NombreDiagnostico del hospital.
    Convierte DxSindrómatico → síntomas en español → predice enfermedad (español)
    y compara directamente con NombreDiagnostico.
    """
    print("=" * 60)
    print("📊 EVALUACIÓN: MODELO DE ENFERMEDAD vs. DIAGNÓSTICOS REALES")
    print("=" * 60)

    mapped_dx = set(SINDROMATICO_A_SINTOMAS.keys())
    df_eval   = df_urgencias[df_urgencias['DxSindromatico'].isin(mapped_dx)].copy()

    if len(df_eval) > sample_size:
        df_eval = df_eval.sample(n=sample_size, random_state=42)

    print(f"   Registros a evaluar: {len(df_eval)}")

    results = []
    for _, row in df_eval.iterrows():
        dx       = row['DxSindromatico']
        sintomas = SINDROMATICO_A_SINTOMAS.get(dx, [])

        if not sintomas:
            continue

        predictions = pipeline.predict_enfermedad(sintomas)
        real_dx     = row['NombreDiagnostico'].lower()

        # Comparar directamente en español (ambas fuentes ya están en español)
        match = any(
            pred['enfermedad'].lower() in real_dx or real_dx in pred['enfermedad'].lower()
            for pred in predictions
        )

        results.append({
            'dx_sindromatico':  dx,
            'real_diagnostico': row['NombreDiagnostico'],
            'predicted_top1':   predictions[0]['enfermedad'],
            'confidence_top1':  predictions[0]['confidence'],
            'match':            match,
            'triaje_real':      row['Triaje'],
        })

    df_results  = pd.DataFrame(results)
    match_rate  = df_results['match'].mean() * 100 if len(df_results) > 0 else 0
    print(f"\n📊 Resultados de la evaluación:")
    print(f"   Tasa de coincidencia (top-3): {match_rate:.1f}%")
    print(f"   Total evaluados: {len(df_results)}")
    print(f"   Coincidencias:   {df_results['match'].sum()}")

    return df_results


def evaluar_triaje_accuracy(pipeline, df_urgencias, sample_size=1000):
    """Evalúa el modelo de triaje contra los datos reales."""
    print(f"\n{'=' * 60}")
    print("📊 EVALUACIÓN: MODELO DE TRIAJE vs. TRIAJE REAL")
    print("=" * 60)

    mapped_dx = set(SINDROMATICO_A_SINTOMAS.keys())
    df_eval = df_urgencias[df_urgencias['DxSindromatico'].isin(mapped_dx)].copy()

    if len(df_eval) > sample_size:
        df_eval = df_eval.sample(n=sample_size, random_state=42)

    print(f"   Registros a evaluar: {len(df_eval)}")

    correct = 0
    y_true = []
    y_pred = []

    for _, row in df_eval.iterrows():
        try:
            result = pipeline.predict_triaje(
                dx_sindromatico=row['DxSindromatico'],
                sexo=row['Sexo'],
                edad=row['Edad'],
                grupo_etario=row['GrupoEtario1'],
                unidad=row['Unidad']
            )
            pred_triaje = result['triaje']
            real_triaje = row['Triage']  # columna real en el Excel

            y_true.append(real_triaje)
            y_pred.append(pred_triaje)

            if pred_triaje == real_triaje:
                correct += 1
        except Exception as e:
            continue

    accuracy = correct / len(y_true) if y_true else 0
    print(f"\n📊 Resultados:")
    print(f"   Accuracy: {accuracy:.4f} ({correct}/{len(y_true)})")

    # Diferencia promedio
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    diff = np.abs(y_true_arr - y_pred_arr)
    print(f"   Error medio absoluto: {diff.mean():.2f} niveles")
    print(f"   Predicciones exactas: {(diff == 0).sum()} ({(diff == 0).mean()*100:.1f}%)")
    print(f"   Error de ±1 nivel: {(diff <= 1).sum()} ({(diff <= 1).mean()*100:.1f}%)")

    return y_true, y_pred


def plot_evaluacion_resultados(df_results, y_true_triaje, y_pred_triaje):
    """Genera visualizaciones de la evaluación."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Tasa de coincidencia por DxSindromatico
    if len(df_results) > 0:
        dx_match = df_results.groupby('dx_sindromatico')['match'].mean().sort_values(ascending=False)
        top_dx = dx_match.head(15)
        colors = ['#2ecc71' if v > 0.5 else '#e74c3c' for v in top_dx.values]
        axes[0, 0].barh(range(len(top_dx)), top_dx.values * 100, color=colors, edgecolor='white')
        axes[0, 0].set_yticks(range(len(top_dx)))
        axes[0, 0].set_yticklabels([d[:35] for d in top_dx.index], fontsize=8)
        axes[0, 0].set_xlabel('% Coincidencia')
        axes[0, 0].set_title('Tasa de Coincidencia por Diagnóstico Sindrómico', fontweight='bold')
        axes[0, 0].invert_yaxis()

    # 2. Distribución de confianza
    if 'confidence_top1' in df_results.columns:
        axes[0, 1].hist(df_results['confidence_top1'], bins=30, color='#3498db',
                       edgecolor='white', alpha=0.85)
        axes[0, 1].set_xlabel('Confianza de la predicción')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].set_title('Distribución de Confianza del Modelo', fontweight='bold')
        axes[0, 1].axvline(df_results['confidence_top1'].mean(), color='red', linestyle='--',
                          label=f'Media: {df_results["confidence_top1"].mean():.3f}')
        axes[0, 1].legend()

    # 3. Matriz de confusión del triaje
    if y_true_triaje and y_pred_triaje:
        from sklearn.metrics import confusion_matrix
        labels = sorted(set(y_true_triaje + y_pred_triaje))
        cm = confusion_matrix(y_true_triaje, y_pred_triaje, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                   xticklabels=[f'T{l}' for l in labels],
                   yticklabels=[f'T{l}' for l in labels])
        axes[1, 0].set_xlabel('Triaje Predicho')
        axes[1, 0].set_ylabel('Triaje Real')
        axes[1, 0].set_title('Matriz de Confusión - Triaje (Datos Reales)', fontweight='bold')

    # 4. Error en triaje
    if y_true_triaje and y_pred_triaje:
        errors = np.array(y_pred_triaje) - np.array(y_true_triaje)
        axes[1, 1].hist(errors, bins=range(-4, 5), color='#e67e22', edgecolor='white',
                       alpha=0.85, align='left')
        axes[1, 1].set_xlabel('Error (Predicho - Real)')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Distribución del Error en Triaje', fontweight='bold')
        axes[1, 1].axvline(0, color='green', linestyle='--', linewidth=2, label='Sin error')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'evaluation_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Resultados guardados en {OUTPUTS_DIR}/evaluation_results.png")


if __name__ == "__main__":
    print("=" * 60)
    print("🔬 EVALUACIÓN COMPLETA DEL PIPELINE")
    print("=" * 60)

    # Cargar pipeline
    pipeline = MedicalPredictionPipeline()

    # Cargar datos de urgencias
    df_urgencias = pd.read_excel(MORBILIDAD_XLSX)
    df_urgencias = df_urgencias[~df_urgencias['Triage'].isin(TRIAJE_EXCLUIDO)]
    print(f"📂 Dataset de urgencias: {len(df_urgencias):,} registros")

    # Evaluar modelo de enfermedad
    df_enfermedad_results = evaluar_enfermedad_en_datos_hospital(pipeline, df_urgencias)

    # Evaluar modelo de triaje
    y_true, y_pred = evaluar_triaje_accuracy(pipeline, df_urgencias)

    # Visualizar
    plot_evaluacion_resultados(df_enfermedad_results, y_true, y_pred)

    # Guardar resultados
    df_enfermedad_results.to_csv(os.path.join(OUTPUTS_DIR, 'enfermedad_evaluation_results.csv'), index=False)
    print(f"\n💾 Resultados detallados guardados en {OUTPUTS_DIR}/")

    print(f"\n{'=' * 60}")
    print(f"✅ EVALUACIÓN COMPLETA")
    print(f"{'=' * 60}")
