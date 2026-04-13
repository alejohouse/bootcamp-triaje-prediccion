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
    TRIAGE_EXCLUDE, SINDROMATIC_TO_SYMPTOMS, DISEASE_TO_DIAGNOSTICO
)
from src.pipeline import MedicalPredictionPipeline
from src.data_preparation import dx_to_symptom_vector


def evaluate_disease_on_hospital_data(pipeline, df_urgencias, sample_size=500):
    """
    Evalúa el modelo de enfermedad contra NombreDiagnostico del hospital.

    Para cada registro del hospital:
    1. Toma el DxSindromatico
    2. Lo convierte a síntomas
    3. Predice la enfermedad
    4. Compara con NombreDiagnostico real
    """
    print("=" * 60)
    print("📊 EVALUACIÓN: MODELO DE ENFERMEDAD vs. DIAGNÓSTICOS REALES")
    print("=" * 60)

    # Filtrar registros con DxSindromatico mapeado
    mapped_dx = set(SINDROMATIC_TO_SYMPTOMS.keys())
    df_eval = df_urgencias[df_urgencias['DxSindromatico'].isin(mapped_dx)].copy()

    if len(df_eval) > sample_size:
        df_eval = df_eval.sample(n=sample_size, random_state=42)

    print(f"   Registros a evaluar: {len(df_eval)}")

    results = []
    for _, row in df_eval.iterrows():
        dx = row['DxSindromatico']
        symptoms = SINDROMATIC_TO_SYMPTOMS.get(dx, [])

        if not symptoms:
            continue

        # Predecir
        predictions = pipeline.predict_disease(symptoms)
        real_dx = row['NombreDiagnostico']

        # Verificar coincidencia
        match = False
        for pred in predictions:
            if pred['disease_es'] != "No mapeado":
                if pred['disease_es'].lower() in real_dx.lower():
                    match = True
                    break

        results.append({
            'dx_sindromatico': dx,
            'real_diagnostico': real_dx,
            'predicted_top1': predictions[0]['disease_en'],
            'predicted_top1_es': predictions[0]['disease_es'],
            'confidence_top1': predictions[0]['confidence'],
            'match': match,
            'triage_real': row['Triage'],
        })

    df_results = pd.DataFrame(results)

    # Métricas
    match_rate = df_results['match'].mean() * 100
    print(f"\n📊 Resultados de la evaluación:")
    print(f"   Tasa de coincidencia (top-3): {match_rate:.1f}%")
    print(f"   Total evaluados: {len(df_results)}")
    print(f"   Coincidencias: {df_results['match'].sum()}")
    print(f"   No coincidentes: {(~df_results['match']).sum()}")

    return df_results


def evaluate_triage_accuracy(pipeline, df_urgencias, sample_size=1000):
    """Evalúa el modelo de triaje contra los datos reales."""
    print(f"\n{'=' * 60}")
    print("📊 EVALUACIÓN: MODELO DE TRIAJE vs. TRIAJE REAL")
    print("=" * 60)

    mapped_dx = set(SINDROMATIC_TO_SYMPTOMS.keys())
    df_eval = df_urgencias[df_urgencias['DxSindromatico'].isin(mapped_dx)].copy()

    if len(df_eval) > sample_size:
        df_eval = df_eval.sample(n=sample_size, random_state=42)

    print(f"   Registros a evaluar: {len(df_eval)}")

    correct = 0
    y_true = []
    y_pred = []

    for _, row in df_eval.iterrows():
        try:
            result = pipeline.predict_triage(
                dx_sindromatico=row['DxSindromatico'],
                sexo=row['Sexo'],
                edad=row['Edad'],
                grupo_etario=row['GrupoEtario1'],
                unidad=row['Unidad']
            )
            pred_triage = result['triage']
            real_triage = row['Triage']

            y_true.append(real_triage)
            y_pred.append(pred_triage)

            if pred_triage == real_triage:
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


def plot_evaluation_results(df_results, y_true_triage, y_pred_triage):
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
    if y_true_triage and y_pred_triage:
        from sklearn.metrics import confusion_matrix
        labels = sorted(set(y_true_triage + y_pred_triage))
        cm = confusion_matrix(y_true_triage, y_pred_triage, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                   xticklabels=[f'T{l}' for l in labels],
                   yticklabels=[f'T{l}' for l in labels])
        axes[1, 0].set_xlabel('Triaje Predicho')
        axes[1, 0].set_ylabel('Triaje Real')
        axes[1, 0].set_title('Matriz de Confusión - Triaje (Datos Reales)', fontweight='bold')

    # 4. Error en triaje
    if y_true_triage and y_pred_triage:
        errors = np.array(y_pred_triage) - np.array(y_true_triage)
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
    df_urgencias = df_urgencias[~df_urgencias['Triage'].isin(TRIAGE_EXCLUDE)]
    print(f"📂 Dataset de urgencias: {len(df_urgencias):,} registros")

    # Evaluar modelo de enfermedad
    df_disease_results = evaluate_disease_on_hospital_data(pipeline, df_urgencias)

    # Evaluar modelo de triaje
    y_true, y_pred = evaluate_triage_accuracy(pipeline, df_urgencias)

    # Visualizar
    plot_evaluation_results(df_disease_results, y_true, y_pred)

    # Guardar resultados
    df_disease_results.to_csv(os.path.join(OUTPUTS_DIR, 'disease_evaluation_results.csv'), index=False)
    print(f"\n💾 Resultados detallados guardados en {OUTPUTS_DIR}/")

    print(f"\n{'=' * 60}")
    print(f"✅ EVALUACIÓN COMPLETA")
    print(f"{'=' * 60}")
