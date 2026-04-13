"""
Preparación de datos para los modelos de predicción de enfermedad y triaje.
Incluye limpieza, encoding, split estratificado y balanceo de clases.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DISEASES_CSV, MORBILIDAD_XLSX, MODELS_DIR, OUTPUTS_DIR,
    RANDOM_STATE, TEST_SIZE, TRIAGE_EXCLUDE, SINDROMATIC_TO_SYMPTOMS
)


def load_diseases_dataset():
    """Carga y preprocesa el dataset de enfermedades y síntomas."""
    print("📂 Cargando dataset de enfermedades y síntomas...")
    df = pd.read_csv(DISEASES_CSV)
    print(f"   Dimensiones: {df.shape}")

    symptom_cols = df.columns[1:].tolist()

    # Eliminar síntomas con varianza muy baja (< 0.005)
    variances = df[symptom_cols].var()
    low_var_cols = variances[variances < 0.005].index.tolist()
    high_var_cols = [c for c in symptom_cols if c not in low_var_cols]

    print(f"   Síntomas originales: {len(symptom_cols)}")
    print(f"   Síntomas eliminados (varianza < 0.005): {len(low_var_cols)}")
    print(f"   Síntomas útiles: {len(high_var_cols)}")

    return df, high_var_cols, low_var_cols


def load_morbilidad_dataset():
    """Carga y preprocesa el dataset de morbilidad (urgencias)."""
    print("\n📂 Cargando dataset de morbilidad (urgencias)...")
    df = pd.read_excel(MORBILIDAD_XLSX)
    original_len = len(df)

    # Excluir triaje 0 y 5
    df = df[~df['Triage'].isin(TRIAGE_EXCLUDE)].copy()
    print(f"   Registros originales: {original_len:,}")
    print(f"   Registros después de excluir triaje {TRIAGE_EXCLUDE}: {len(df):,}")

    return df


def encode_morbilidad_features(df):
    """Codifica variables categóricas del dataset de morbilidad."""
    print("\n🔧 Codificando variables categóricas...")
    df_encoded = df.copy()

    # Label encoding para columnas categóricas
    encoders = {}
    categorical_cols = ['DxSindromatico', 'Sexo', 'GrupoEtario1', 'Unidad']

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
        print(f"   {col}: {len(le.classes_)} categorías codificadas")

    # Guardar encoders
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(encoders, os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    print(f"   💾 Encoders guardados en {MODELS_DIR}/label_encoders.pkl")

    return df_encoded, encoders


def prepare_disease_data(df, symptom_cols):
    """Prepara datos para el modelo de predicción de enfermedad."""
    print("\n📊 Preparando datos para el modelo de ENFERMEDAD...")

    # Filtrar enfermedades con menos de 5 muestras (no se pueden estratificar)
    disease_counts = df['diseases'].value_counts()
    valid_diseases = disease_counts[disease_counts >= 5].index
    df_filtered = df[df['diseases'].isin(valid_diseases)].copy()
    removed = len(df) - len(df_filtered)
    print(f"   Enfermedades con <5 muestras eliminadas: {len(disease_counts) - len(valid_diseases)} ({removed} registros)")
    print(f"   Enfermedades válidas: {len(valid_diseases)}")

    X = df_filtered[symptom_cols].values
    y = df_filtered['diseases'].values

    # Label encoding para enfermedades
    le_disease = LabelEncoder()
    y_encoded = le_disease.fit_transform(y)

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"   Clases: {len(le_disease.classes_)}")

    # Guardar encoder
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(le_disease, os.path.join(MODELS_DIR, 'disease_label_encoder.pkl'))

    return X_train, X_test, y_train, y_test, le_disease


def prepare_triage_data(df_encoded):
    """Prepara datos para el modelo de predicción de triaje."""
    print("\n📊 Preparando datos para el modelo de TRIAJE...")

    feature_cols = [
        'DxSindromatico_encoded', 'Sexo_encoded', 'Edad',
        'GrupoEtario1_encoded', 'Unidad_encoded'
    ]

    X = df_encoded[feature_cols].values
    y = df_encoded['Triage'].values

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"   Distribución triaje (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"     Triaje {u}: {c:,} ({c/len(y_train)*100:.1f}%)")

    # Aplicar SMOTE para balancear
    print("\n   🔄 Aplicando SMOTE + Tomek Links para balanceo...")
    smote_tomek = SMOTETomek(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train)

    print(f"   X_train después de balanceo: {X_train_bal.shape}")
    print(f"   Distribución triaje (balanceado):")
    unique_bal, counts_bal = np.unique(y_train_bal, return_counts=True)
    for u, c in zip(unique_bal, counts_bal):
        print(f"     Triaje {u}: {c:,} ({c/len(y_train_bal)*100:.1f}%)")

    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    # Guardar scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'triage_scaler.pkl'))

    return X_train_scaled, X_test_scaled, y_train_bal, y_test, scaler


def dx_to_symptom_vector(dx_sindromatico, symptom_columns):
    """Convierte un DxSindromatico a un vector binario de síntomas."""
    vector = np.zeros(len(symptom_columns))

    if dx_sindromatico in SINDROMATIC_TO_SYMPTOMS:
        symptoms = SINDROMATIC_TO_SYMPTOMS[dx_sindromatico]
        for symptom in symptoms:
            if symptom in symptom_columns:
                idx = symptom_columns.index(symptom)
                vector[idx] = 1

    return vector


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 PREPARACIÓN DE DATOS")
    print("=" * 60)

    # 1. Dataset de enfermedades
    df_diseases, symptom_cols, low_var = load_diseases_dataset()
    X_train_d, X_test_d, y_train_d, y_test_d, le_disease = prepare_disease_data(
        df_diseases, symptom_cols
    )

    # Guardar columnas de síntomas útiles
    joblib.dump(symptom_cols, os.path.join(MODELS_DIR, 'symptom_columns.pkl'))

    # 2. Dataset de morbilidad
    df_morb = load_morbilidad_dataset()
    df_morb_encoded, encoders = encode_morbilidad_features(df_morb)
    X_train_t, X_test_t, y_train_t, y_test_t, scaler = prepare_triage_data(df_morb_encoded)

    # Guardar datos preparados
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    np.savez(
        os.path.join(OUTPUTS_DIR, 'disease_data.npz'),
        X_train=X_train_d, X_test=X_test_d,
        y_train=y_train_d, y_test=y_test_d
    )
    np.savez(
        os.path.join(OUTPUTS_DIR, 'triage_data.npz'),
        X_train=X_train_t, X_test=X_test_t,
        y_train=y_train_t, y_test=y_test_t
    )

    print("\n" + "=" * 60)
    print("✅ DATOS PREPARADOS Y GUARDADOS")
    print("=" * 60)
