"""
Pipeline unificado de predicción.
Recibe síntomas del paciente (en español) y predice:
1. La enfermedad probable (Top-3)
2. El nivel de triaje recomendado
"""
import numpy as np
import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODELS_DIR, SINDROMATICO_A_SINTOMAS, TRIAJE_LABELS


class MedicalPredictionPipeline:
    """Pipeline unificado para predicción médica en español."""

    def __init__(self):
        """Carga todos los modelos y encoders necesarios."""
        print("🔄 Cargando modelos y configuración...")

        # Modelo de enfermedad
        self.enfermedad_model   = joblib.load(os.path.join(MODELS_DIR, 'enfermedad_model.pkl'))
        self.enfermedad_encoder = joblib.load(os.path.join(MODELS_DIR, 'enfermedad_label_encoder.pkl'))
        self.sintomas_columns = joblib.load(os.path.join(MODELS_DIR, 'sintomas_columns.pkl'))

        # Scaler opcional del modelo de enfermedad (solo si ganó Logistic Regression)
        scaler_path = os.path.join(MODELS_DIR, 'enfermedad_scaler.pkl')
        self.enfermedad_scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        # Modelo de triaje
        self.triaje_model   = joblib.load(os.path.join(MODELS_DIR, 'triaje_model.pkl'))
        self.triaje_scaler  = joblib.load(os.path.join(MODELS_DIR, 'triaje_scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))

        print("✅ Pipeline cargado exitosamente")

    def predict_enfermedad(self, sintomas_list):
        """
        Predice la enfermedad basándose en una lista de síntomas en español.

        Args:
            sintomas_list: Lista de síntomas en español
                          ej: ['dolor de cabeza', 'fiebre', 'náuseas']

        Returns:
            lista de dicts con top-3 predicciones y confianza
        """
        # Crear vector binario
        vector = np.zeros(len(self.sintomas_columns))
        for symptom in sintomas_list:
            if symptom in self.sintomas_columns:
                idx = self.sintomas_columns.index(symptom)
                vector[idx] = 1

        X = vector.reshape(1, -1)
        if self.enfermedad_scaler is not None:
            X = self.enfermedad_scaler.transform(X)

        if hasattr(self.enfermedad_model, 'predict_proba'):
            probas      = self.enfermedad_model.predict_proba(X)[0]
            top_indices = np.argsort(probas)[-3:][::-1]

            predictions = []
            for idx in top_indices:
                enfermedad    = self.enfermedad_encoder.inverse_transform([idx])[0]
                confidence = float(probas[idx])
                predictions.append({
                    'enfermedad':    enfermedad,
                    'confidence': confidence,
                })
        else:
            pred    = self.enfermedad_model.predict(X)[0]
            enfermedad = self.enfermedad_encoder.inverse_transform([pred])[0]
            predictions = [{'enfermedad': enfermedad, 'confidence': 1.0}]

        return predictions

    def predict_triaje(self, dx_sindromatico, sexo, edad, grupo_etario, unidad):
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
            dx_enc     = self.label_encoders['DxSindromatico'].transform([dx_sindromatico])[0]
            sexo_enc   = self.label_encoders['Sexo'].transform([sexo])[0]
            grupo_enc  = self.label_encoders['GrupoEtario1'].transform([grupo_etario])[0]
            unidad_enc = self.label_encoders['Unidad'].transform([unidad])[0]
        except ValueError as e:
            print(f"⚠️ Valor no reconocido: {e}")
            return {'triaje': 3, 'description': TRIAJE_LABELS.get(3, 'Urgencia'),
                    'warning': str(e)}

        X        = np.array([[dx_enc, sexo_enc, edad, grupo_enc, unidad_enc]])
        X_scaled = self.triaje_scaler.transform(X)

        triaje_pred = self.triaje_model.predict(X_scaled)[0]

        return {
            'triaje':      int(triaje_pred),
            'descripción': TRIAJE_LABELS.get(int(triaje_pred), f'Nivel {triaje_pred}')
        }

    def predict_full(self, sintomas_list, sexo='Masculino', edad=30,
                     grupo_etario='Entre 15 y 44',
                     unidad='URGENCIAS CONSULTA Y PROCEDIMIENTOS'):
        """
        Pipeline completo: síntomas → enfermedad + triaje.

        Args:
            symptoms_list: Lista de síntomas en español
            sexo, edad, grupo_etario, unidad: Datos demográficos

        Returns:
            dict con predicción completa
        """
        enfermedad_predictions = self.predict_enfermedad(sintomas_list)
        best_dx             = self._find_closest_dx(sintomas_list)
        triaje_result       = self.predict_triaje(best_dx, sexo, edad, grupo_etario, unidad)

        return {
            'sintomas_input':      sintomas_list,
            'enfermedad_predictions': enfermedad_predictions,
            'dx_sindromatico':     best_dx,
            'triaje':              triaje_result,
            'patient_info': {
                'sexo':         sexo,
                'edad':         edad,
                'grupo_etario': grupo_etario,
                'unidad':       unidad,
            }
        }

    def _find_closest_dx(self, sintomas_list):
        """Encuentra el DxSindromatico más cercano a los síntomas (similaridad Jaccard)."""
        best_score = -1
        best_dx    = "DOLOR NO ESPECIFICADO"

        for dx, dx_symptoms in SINDROMATICO_A_SINTOMAS.items():
            input_set = set(sintomas_list)
            dx_set    = set(dx_symptoms)
            if len(input_set | dx_set) > 0:
                score = len(input_set & dx_set) / len(input_set | dx_set)
                if score > best_score:
                    best_score = score
                    best_dx    = dx

        return best_dx

    def get_all_symptoms(self):
        """Retorna la lista completa de síntomas disponibles (en español)."""
        return self.sintomas_columns

    def get_all_dx_sindromatico(self):
        """Retorna todos los diagnósticos sindrómicos mapeados."""
        return list(SINDROMATICO_A_SINTOMAS.keys())


if __name__ == "__main__":
    print("=" * 60)
    print("🔬 TEST DEL PIPELINE DE PREDICCIÓN")
    print("=" * 60)

    pipeline = MedicalPredictionPipeline()

    print("\n📋 Test 1: Dolor abdominal")
    result = pipeline.predict_full(
        sintomas_list=['dolor abdominal agudo', 'náuseas', 'vómitos', 'fiebre'],
        sexo='Femenino', edad=35
    )
    print(f"   DxSindromatico: {result['dx_sindromatico']}")
    print(f"   Triaje: {result['triaje']}")
    for p in result['enfermedad_predictions']:
        print(f"     - {p['enfermedad']} ({p['confidence']*100:.1f}%)")

    print("\n📋 Test 2: Síntomas respiratorios")
    result2 = pipeline.predict_full(
        sintomas_list=['toser expectorando', 'fiebre', 'dificultad para respirar', 'dolor de garganta'],
        sexo='Masculino', edad=50
    )
    print(f"   DxSindromatico: {result2['dx_sindromatico']}")
    print(f"   Triaje: {result2['triaje']}")
    for p in result2['enfermedad_predictions']:
        print(f"     - {p['enfermedad']} ({p['confidence']*100:.1f}%)")

    print(f"\n{'=' * 60}")
    print(f"✅ PIPELINE FUNCIONAL")
    print(f"{'=' * 60}")
