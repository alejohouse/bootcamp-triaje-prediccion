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
    ENFERMEDAD_XLSX, MORBILIDAD_XLSX, MODELS_DIR, OUTPUTS_DIR,
    RANDOM_STATE, TEST_SIZE, TRIAJE_EXCLUIDO, SINDROMATICO_A_SINTOMAS
)


def load_enfermedad_dataset():
    """Carga y preprocesa el dataset de enfermedades y síntomas (xlsx en español)."""
    print("📂 Cargando dataset de enfermedades y síntomas...")
    df = pd.read_excel(ENFERMEDAD_XLSX)
    print(f"   Dimensiones: {df.shape}")

    # Normalizar nombres de enfermedades (evitar duplicados por capitalización)
    df['enfermedad'] = df['enfermedad'].str.strip().str.lower()

    sintoma_cols = df.columns[1:].tolist()

    # Eliminar síntomas con varianza muy baja (< 0.005)
    varianza = df[sintoma_cols].var()
    var_baja_cols = varianza[varianza < 0.005].index.tolist()
    var_alta_cols = [c for c in sintoma_cols if c not in var_baja_cols]

    print(f"   Síntomas originales: {len(sintoma_cols)}")
    print(f"   Síntomas eliminados (varianza < 0.005): {len(var_baja_cols)}")
    print(f"   Síntomas útiles: {len(var_alta_cols)}")

    return df, var_alta_cols, var_baja_cols


def load_morbilidad_dataset():
    """Carga y preprocesa el dataset de morbilidad (urgencias)."""
    print("\n📂 Cargando dataset de morbilidad (urgencias)...")
    df = pd.read_excel(MORBILIDAD_XLSX)
    original_len = len(df)

    # Excluir triaje 0 y 5
    df = df[~df['Triage'].isin(TRIAJE_EXCLUIDO)].copy()
    print(f"   Registros originales: {original_len:,}")
    print(f"   Registros después de excluir triaje {TRIAJE_EXCLUIDO}: {len(df):,}")

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


def prepare_enfermedad_data(df, sintoma_cols):
    """Prepara datos para el modelo de predicción de enfermedad."""
    print("\n📊 Preparando datos para el modelo de ENFERMEDAD...")

    # Filtrar enfermedades con menos de 5 muestras
    enfermedad_counts = df['enfermedad'].value_counts()
    valid_enfermedades = enfermedad_counts[enfermedad_counts >= 5].index
    df_filtered = df[df['enfermedad'].isin(valid_enfermedades)].copy()
    removed = len(df) - len(df_filtered)
    print(f"   Enfermedades con <5 muestras eliminadas: {len(enfermedad_counts) - len(valid_enfermedades)} ({removed} registros)")
    print(f"   Enfermedades válidas: {len(valid_enfermedades)}")

    X = df_filtered[sintoma_cols].values
    y = df_filtered['enfermedad'].values

    # Label encoding para enfermedades
    le_enfermedad = LabelEncoder()
    y_encoded = le_enfermedad.fit_transform(y)

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"   Clases: {len(le_enfermedad.classes_)}")

    # Aplicar SMOTE para balancear clases desbalanceadas
    print("\n   🔄 Aplicando SMOTE para balanceo de enfermedades...")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"   X_train después de SMOTE: {X_train_bal.shape}")

    # Guardar encoder
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(le_enfermedad, os.path.join(MODELS_DIR, 'enfermedad_label_encoder.pkl'))

    return X_train_bal, X_test, y_train_bal, y_test, le_enfermedad


def prepare_triaje_data(df_encoded):
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
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'triaje_scaler.pkl'))

    return X_train_scaled, X_test_scaled, y_train_bal, y_test, scaler


def dx_to_sintoma_vector(dx_sindromatico, sintoma_columns):
    """Convierte un DxSindromatico a un vector binario de síntomas."""
    vector = np.zeros(len(sintoma_columns))

    if dx_sindromatico in SINDROMATICO_A_SINTOMAS:
        sintomas = SINDROMATICO_A_SINTOMAS[dx_sindromatico]
        for sintoma in sintomas:
            if sintoma in sintoma_columns:
                idx = sintoma_columns.index(sintoma)
                vector[idx] = 1

    return vector


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 PREPARACIÓN DE DATOS")
    print("=" * 60)

    # 1. Dataset de enfermedades
    df_enfermedades, sintoma_cols, baja_var = load_enfermedad_dataset()
    X_train_e, X_test_e, y_train_e, y_test_e, le_enfermedad = prepare_enfermedad_data(
        df_enfermedades, sintoma_cols
    )

    # Guardar columnas de síntomas útiles y encoder
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(sintoma_cols, os.path.join(MODELS_DIR, 'sintoma_columns.pkl'))
    print(f"   💾 symptom_columns guardado en {MODELS_DIR}/sintoma_columns.pkl")

    # 2. Dataset de morbilidad
    df_morb = load_morbilidad_dataset()
    df_morb_encoded, encoders = encode_morbilidad_features(df_morb)
    X_train_t, X_test_t, y_train_t, y_test_t, scaler = prepare_triaje_data(df_morb_encoded)

    print("\n" + "=" * 60)
    print("✅ DATOS PREPARADOS Y GUARDADOS")
    print("=" * 60)
