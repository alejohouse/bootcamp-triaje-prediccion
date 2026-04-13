"""
Configuración central del proyecto de predicción de enfermedades y triaje.
"""
import os

# ============================================================
# RUTAS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSUMOS_DIR = os.path.join(BASE_DIR, "insumos")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Archivos de entrada
DISEASES_CSV = os.path.join(INSUMOS_DIR, "Final_Augmented_dataset_Diseases_and_Symptoms.csv")
MORBILIDAD_XLSX = os.path.join(INSUMOS_DIR, "Morbilidad_urgencias_Hospital_Pitalito_20260406.xlsx")

# ============================================================
# CONSTANTES DEL MODELO
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Niveles de triaje a excluir (muy pocas muestras: 0=1 caso, 5=23 casos)
TRIAGE_EXCLUDE = [0, 5]
TRIAGE_LABELS = {1: "Resucitación", 2: "Emergencia", 3: "Urgencia", 4: "Menos urgente"}

# ============================================================
# DICCIONARIO DE MAPEO: DxSindromatico → Síntomas en inglés
# ============================================================
# Mapeo de los diagnósticos sindrómicos más comunes del hospital
# a los nombres de síntomas del dataset de enfermedades
SINDROMATIC_TO_SYMPTOMS = {
    "DOLOR ABDOMINAL CONSTANTE": ["sharp abdominal pain", "abdominal distention", "nausea", "vomiting"],
    "DOLOR NO ESPECIFICADO": ["pain of the anus", "sharp chest pain", "sharp abdominal pain", "ache all over"],
    "CEFALEA": ["headache", "dizziness", "nausea", "vomiting"],
    "DOLOR ABDOMINAL": ["sharp abdominal pain", "abdominal distention", "nausea"],
    "DOLOR AGUDO": ["sharp chest pain", "sharp abdominal pain", "ache all over"],
    "HERIDA DE OTRA PARTE NO ESPECIFICADA": ["skin lesion", "bleeding or discharge from wound", "swollen lymph nodes"],
    "CEFALEA Y FIEBRE": ["headache", "fever", "nausea", "vomiting", "ache all over"],
    "DIARREA SIN COMPROMISO GENERAL": ["diarrhea", "nausea", "abdominal distention", "flatulence"],
    "CONTUSIONES": ["skin lesion", "swelling", "ache all over"],
    "SINDROME FEBRIL": ["fever", "chills", "ache all over", "headache"],
    "ACCIDENTE DE TRANSITO": ["skin lesion", "bleeding or discharge from wound", "sharp chest pain", "dizziness"],
    "TOS CON EXPECTORACION Y FIEBRE": ["cough", "fever", "sore throat", "shortness of breath", "pus in sputum"],
    "TOS SECA MALESTAR GENERAL FIEBRE": ["cough", "fever", "ache all over", "sore throat"],
    "DOLOR ABDOMINAL CON DIARREA Y DHT": ["sharp abdominal pain", "diarrhea", "nausea", "vomiting", "fever"],
    "DOLOR ABDOMINAL INTERMITENTE": ["sharp abdominal pain", "nausea", "abdominal distention"],
    "DOLOR TORAXICO": ["sharp chest pain", "chest tightness", "shortness of breath", "palpitations"],
    "CELULITIS DE SITIO NO ESPECIFICADO": ["skin lesion", "skin rash", "fever", "swollen lymph nodes"],
    "INFECCION VIAS URINARIAS": ["painful urination", "frequent urination", "suprapubic pain", "fever"],
    "DOLOR ABDOMINAL SEVERO": ["sharp abdominal pain", "nausea", "vomiting", "abdominal distention", "fever"],
    "DISNEA MODERADA": ["shortness of breath", "breathing fast", "chest tightness", "cough"],
    "DOLOR TORACICO": ["sharp chest pain", "chest tightness", "shortness of breath"],
    "LESIONES POR VIOLENCIA": ["skin lesion", "bleeding or discharge from wound", "swelling"],
    "RETENCION URINARIA": ["retention of urine", "suprapubic pain", "painful urination"],
    "TRAUMATISMO CRANEOENCEFALICO": ["headache", "dizziness", "nausea", "vomiting", "fainting"],
    "DOLOR ABDOMINAL INTERMITENTE CON NAUSEAS": ["sharp abdominal pain", "nausea", "vomiting"],
    "ESQUIZOFRENIA CON SINTOMAS DE ALUCINACIONES": ["depressive or psychotic symptoms", "hostile behavior", "insomnia", "anxiety and nervousness"],
    "DOLOR LUMBAR": ["low back pain", "back pain", "ache all over"],
    "DOLOR DE OIDO": ["ear pain", "diminished hearing", "fever"],
    "DOLOR DE GARGANTA": ["sore throat", "throat swelling", "difficulty in swallowing", "hoarse voice"],
    "VOMITO": ["vomiting", "nausea", "abdominal distention"],
    "MAREO": ["dizziness", "nausea", "fainting", "headache"],
    "CRISIS HIPERTENSIVA": ["sharp chest pain", "headache", "dizziness", "shortness of breath"],
    "FRACTURA": ["arm pain", "leg pain", "swelling", "skin lesion"],
    "COLICO RENAL": ["sharp abdominal pain", "painful urination", "nausea", "vomiting"],
    "QUEMADURA": ["skin lesion", "skin rash", "fever"],
    "CRISIS CONVULSIVA": ["seizures", "fainting", "dizziness", "abnormal involuntary movements"],
    "MORDEDURA DE ANIMAL": ["skin lesion", "bleeding or discharge from wound", "swollen lymph nodes", "fever"],
    "DOLOR EN EXTREMIDADES": ["arm pain", "leg pain", "ache all over", "swelling"],
    "EPISTAXIS": ["nosebleed", "dizziness", "headache"],
    "DOLOR DENTAL": ["toothache", "jaw pain", "swelling"],
}

# ============================================================
# MAPEO: Enfermedades inglés → NombreDiagnostico español (CIE-10)
# ============================================================
DISEASE_TO_DIAGNOSTICO = {
    "pneumonia": "NEUMONIA",
    "urinary tract infection": "INFECCION DE VIAS URINARIAS",
    "acute bronchitis": "BRONQUITIS AGUDA",
    "gastroenteritis": "DIARREA GASTROENTERITIS",
    "infectious gastroenteritis": "DIARREA GASTROENTERITIS PRESUNTO ORIGEN INFECCIOSO",
    "migraine": "MIGRAÑA",
    "anxiety": "TRASTORNO DE ANSIEDAD",
    "hypertension": "HIPERTENSION ESENCIAL",
    "asthma": "ASMA",
    "appendicitis": "APENDICITIS",
    "fracture": "FRACTURA",
    "diabetes": "DIABETES MELLITUS",
    "allergic reaction": "REACCION ALERGICA",
    "otitis media": "OTITIS MEDIA",
    "sinusitis": "SINUSITIS",
    "cellulitis": "CELULITIS",
    "renal colic": "COLICO RENAL",
    "epilepsy": "EPILEPSIA",
    "gout": "GOTA",
    "conjunctivitis": "CONJUNTIVITIS",
}

# ============================================================
# PARÁMETROS DE MODELOS
# ============================================================
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
}
