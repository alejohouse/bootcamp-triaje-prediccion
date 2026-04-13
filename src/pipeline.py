"""
Pipeline unificado de predicción.
Recibe síntomas del paciente y predice:
1. La enfermedad probable (Top-3)
2. El nivel de triaje recomendado
3. Comparación con NombreDiagnostico real (cuando disponible)
"""
import numpy as np
import pandas as pd
import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MODELS_DIR, SINDROMATIC_TO_SYMPTOMS, DISEASE_TO_DIAGNOSTICO, TRIAGE_LABELS
)


class MedicalPredictionPipeline:
    """Pipeline unificado para predicción médica."""

    def __init__(self):
        """Carga todos los modelos y encoders necesarios."""
        print("🔄 Cargando modelos y configuración...")

        # Modelo de enfermedad
        self.disease_model = joblib.load(os.path.join(MODELS_DIR, 'disease_model.pkl'))
        self.disease_encoder = joblib.load(os.path.join(MODELS_DIR, 'disease_label_encoder.pkl'))
        self.symptom_columns = joblib.load(os.path.join(MODELS_DIR, 'symptom_columns.pkl'))

        # Modelo de triaje
        self.triage_model = joblib.load(os.path.join(MODELS_DIR, 'triage_model.pkl'))
        self.triage_scaler = joblib.load(os.path.join(MODELS_DIR, 'triage_scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))

        # Mapeo inverso para DxSindromatico
        self._symptoms_to_dx = {}
        for dx, symptoms in SINDROMATIC_TO_SYMPTOMS.items():
            key = frozenset(symptoms)
            self._symptoms_to_dx[key] = dx

        print("✅ Pipeline cargado exitosamente")

    def predict_disease(self, symptoms_list):
        """
        Predice la enfermedad basándose en una lista de síntomas.

        Args:
            symptoms_list: Lista de nombres de síntomas en inglés
                          ej: ['headache', 'fever', 'nausea']

        Returns:
            dict con top-3 predicciones y confianza
        """
        # Crear vector binario
        vector = np.zeros(len(self.symptom_columns))
        for symptom in symptoms_list:
            if symptom in self.symptom_columns:
                idx = self.symptom_columns.index(symptom)
                vector[idx] = 1

        # Predecir
        X = vector.reshape(1, -1)

        if hasattr(self.disease_model, 'predict_proba'):
            probas = self.disease_model.predict_proba(X)[0]
            top_indices = np.argsort(probas)[-3:][::-1]

            predictions = []
            for idx in top_indices:
                disease = self.disease_encoder.inverse_transform([idx])[0]
                confidence = probas[idx]
                diagnostico_es = DISEASE_TO_DIAGNOSTICO.get(disease, "No mapeado")
                predictions.append({
                    'disease_en': disease,
                    'disease_es': diagnostico_es,
                    'confidence': float(confidence),
                })
        else:
            pred = self.disease_model.predict(X)[0]
            disease = self.disease_encoder.inverse_transform([pred])[0]
            diagnostico_es = DISEASE_TO_DIAGNOSTICO.get(disease, "No mapeado")
            predictions = [{
                'disease_en': disease,
                'disease_es': diagnostico_es,
                'confidence': 1.0,
            }]

        return predictions

    def predict_triage(self, dx_sindromatico, sexo, edad, grupo_etario, unidad):
        """
        Predice el nivel de triaje.

        Args:
            dx_sindromatico: Diagnóstico sindrómico (español)
            sexo: 'Masculino' o 'Femenino'
            edad: Edad en años
            grupo_etario: Grupo etario (ej: 'Entre 15 y 44')
            unidad: Unidad de atención

        Returns:
            dict con nivel de triaje predicho y descripción
        """
        try:
            # Codificar features
            dx_enc = self.label_encoders['DxSindromatico'].transform([dx_sindromatico])[0]
            sexo_enc = self.label_encoders['Sexo'].transform([sexo])[0]
            grupo_enc = self.label_encoders['GrupoEtario1'].transform([grupo_etario])[0]
            unidad_enc = self.label_encoders['Unidad'].transform([unidad])[0]
        except ValueError as e:
            # Si el valor no fue visto durante entrenamiento, usar uno genérico
            print(f"⚠️ Valor no reconocido: {e}")
            return {'triage': 3, 'description': TRIAGE_LABELS.get(3, 'Urgencia'),
                    'warning': str(e)}

        X = np.array([[dx_enc, sexo_enc, edad, grupo_enc, unidad_enc]])
        X_scaled = self.triage_scaler.transform(X)

        triage_pred = self.triage_model.predict(X_scaled)[0]

        return {
            'triage': int(triage_pred),
            'description': TRIAGE_LABELS.get(int(triage_pred), f'Nivel {triage_pred}')
        }

    def predict_full(self, symptoms_list, sexo='Masculino', edad=30,
                     grupo_etario='Entre 15 y 44',
                     unidad='URGENCIAS CONSULTA Y PROCEDIMIENTOS'):
        """
        Pipeline completo: síntomas → enfermedad + triaje.

        Args:
            symptoms_list: Lista de síntomas en inglés
            sexo, edad, grupo_etario, unidad: Datos demográficos

        Returns:
            dict con predicción completa
        """
        # 1. Predecir enfermedad
        disease_predictions = self.predict_disease(symptoms_list)

        # 2. Buscar DxSindromatico más cercano basado en síntomas
        best_dx = self._find_closest_dx(symptoms_list)

        # 3. Predecir triaje
        triage_result = self.predict_triage(
            best_dx, sexo, edad, grupo_etario, unidad
        )

        return {
            'symptoms_input': symptoms_list,
            'disease_predictions': disease_predictions,
            'dx_sindromatico': best_dx,
            'triage': triage_result,
            'patient_info': {
                'sexo': sexo,
                'edad': edad,
                'grupo_etario': grupo_etario,
                'unidad': unidad,
            }
        }

    def _find_closest_dx(self, symptoms_list):
        """Encuentra el DxSindromatico más cercano a los síntomas dados."""
        best_score = -1
        best_dx = "DOLOR NO ESPECIFICADO"  # Default

        for dx, dx_symptoms in SINDROMATIC_TO_SYMPTOMS.items():
            # Calcular similaridad (Jaccard)
            input_set = set(symptoms_list)
            dx_set = set(dx_symptoms)
            if len(input_set | dx_set) > 0:
                score = len(input_set & dx_set) / len(input_set | dx_set)
                if score > best_score:
                    best_score = score
                    best_dx = dx

        return best_dx

    def compare_with_real(self, prediction, real_diagnostico):
        """
        Compara la predicción con el diagnóstico real del hospital.

        Args:
            prediction: Output de predict_full()
            real_diagnostico: NombreDiagnostico real del dataset

        Returns:
            dict con resultado de la comparación
        """
        real_lower = real_diagnostico.lower()
        matches = []

        for pred in prediction['disease_predictions']:
            disease_es = pred['disease_es'].lower() if pred['disease_es'] != "No mapeado" else ""
            if disease_es and disease_es in real_lower:
                matches.append(pred)

        return {
            'real_diagnostico': real_diagnostico,
            'predicted': prediction['disease_predictions'],
            'matches': matches,
            'match_found': len(matches) > 0,
            'triage_predicted': prediction['triage']['triage'],
        }

    def get_all_symptoms(self):
        """Retorna la lista completa de síntomas disponibles."""
        return self.symptom_columns

    def get_all_dx_sindromatico(self):
        """Retorna todos los diagnósticos sindrómicos mapeados."""
        return list(SINDROMATIC_TO_SYMPTOMS.keys())


if __name__ == "__main__":
    print("=" * 60)
    print("🔬 TEST DEL PIPELINE DE PREDICCIÓN")
    print("=" * 60)

    pipeline = MedicalPredictionPipeline()

    # Test 1: Síntomas de dolor abdominal
    print("\n📋 Test 1: Dolor abdominal")
    result = pipeline.predict_full(
        symptoms_list=['sharp abdominal pain', 'nausea', 'vomiting', 'fever'],
        sexo='Femenino', edad=35
    )
    print(f"   DxSindromatico: {result['dx_sindromatico']}")
    print(f"   Triaje: {result['triage']}")
    print(f"   Top 3 enfermedades:")
    for p in result['disease_predictions']:
        print(f"     - {p['disease_en']} ({p['confidence']*100:.1f}%) → {p['disease_es']}")

    # Test 2: Síntomas respiratorios
    print("\n📋 Test 2: Síntomas respiratorios")
    result2 = pipeline.predict_full(
        symptoms_list=['cough', 'fever', 'shortness of breath', 'sore throat'],
        sexo='Masculino', edad=50
    )
    print(f"   DxSindromatico: {result2['dx_sindromatico']}")
    print(f"   Triaje: {result2['triage']}")
    print(f"   Top 3 enfermedades:")
    for p in result2['disease_predictions']:
        print(f"     - {p['disease_en']} ({p['confidence']*100:.1f}%) → {p['disease_es']}")

    # Test 3: Cefalea
    print("\n📋 Test 3: Cefalea con fiebre")
    result3 = pipeline.predict_full(
        symptoms_list=['headache', 'fever', 'nausea', 'dizziness'],
        sexo='Femenino', edad=28
    )
    print(f"   DxSindromatico: {result3['dx_sindromatico']}")
    print(f"   Triaje: {result3['triage']}")
    print(f"   Top 3 enfermedades:")
    for p in result3['disease_predictions']:
        print(f"     - {p['disease_en']} ({p['confidence']*100:.1f}%) → {p['disease_es']}")

    print(f"\n{'=' * 60}")
    print(f"✅ PIPELINE FUNCIONAL")
    print(f"{'=' * 60}")
