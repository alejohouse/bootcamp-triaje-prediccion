# 🏥 Predictor Médico: Triaje & Diagnóstico con Machine Learning

> Sistema de apoyo clínico que predice las **3 enfermedades más probables** y el **nivel de triaje recomendado** a partir de los síntomas reportados por el paciente.

Desarrollado como proyecto académico del Bootcamp de Ciencia de Datos — Hospital de Pitalito (Huila, Colombia).

---

## 📋 Tabla de contenidos

1. [¿Qué hace este sistema?](#-qué-hace-este-sistema)
2. [Datos utilizados](#-datos-utilizados)
3. [Exploración de datos — Hallazgos clave](#-exploración-de-datos--hallazgos-clave)
4. [Modelos entrenados](#-modelos-entrenados)
5. [Resultados de precisión](#-resultados-de-precisión)
6. [Instalación y configuración](#-instalación-y-configuración)
7. [Cómo usar la aplicación](#-cómo-usar-la-aplicación)
8. [Estructura del proyecto](#-estructura-del-proyecto)
9. [Aviso legal](#️-aviso-legal)

---

## 🔬 ¿Qué hace este sistema?

El **Predictor Médico** combina dos pipelines de Machine Learning:

| Pipeline | Entrada | Salida |
|---|---|---|
| **Predictor de Enfermedad** | Lista de síntomas en español | Top 3 enfermedades con % de confianza |
| **Predictor de Triaje** | Síntomas + sexo + edad + unidad | Nivel de triaje 1–4 con descripción |

El sistema conecta ambos modelos usando una **similitud Jaccard** que identifica el diagnóstico sindrómico del hospital más cercano a los síntomas del paciente, sin necesidad de que el médico lo ingrese manualmente.

---

## 📂 Datos utilizados

El sistema fue entrenado con **dos fuentes de datos**:

### 1. Dataset de Enfermedades y Síntomas
- **Origen:** Dataset público de síntomas vs. enfermedades (en español)
- **Dimensiones originales:** 246,945 registros × 378 columnas
- **Enfermedades representadas:** 773 (721 con suficientes muestras para entrenar)
- **Síntomas totales:** 377 → **213 útiles** después de filtrar por varianza < 0.005
- **Balanceo:** SMOTE (k=3) → 702,975 registros de entrenamiento

> ⚠️ Este archivo pesa 249 MB y **no está incluido en el repositorio**. Debe colocarse en `insumos/sintomas_vs_enfermedades.xlsx` para reentrenar los modelos.

### 2. Registros de Urgencias — Hospital de Pitalito
- **Origen:** Base de datos real de urgencias 2024–2026
- **Dimensiones:** 59,025 registros originales → **59,001 válidos** (excluye triajes 0 y 5 por escasez de muestras)
- **Variables:** DxSindrómico, Sexo, Edad, Grupo Etario, Unidad de atención, Nivel de triaje
- **Balanceo:** SMOTETomek → 126,636 registros de entrenamiento equilibrados

---

## 📊 Exploración de datos — Hallazgos clave

### Distribución del triaje en urgencias

| Nivel | Descripción | Registros | % |
|---|---|---|---|
| 🔴 **1** | Resucitación | 106 | 0.2% |
| 🟠 **2** | Emergencia | 3,577 | 7.6% |
| 🟡 **3** | Urgencia | 31,764 | 67.3% |
| 🟢 **4** | Menos urgente | 11,753 | 24.9% |

> El **67% de los pacientes** ingresan con triaje 3 (Urgencia). El desbalance extremo en triaje 1 motivó el uso de SMOTETomek.

### Síntomas — Dataset de enfermedades

- **164 síntomas eliminados** por baja varianza (< 0.005): no discriminan entre enfermedades.
- **52 enfermedades excluidas** por tener menos de 5 muestras (no permiten split estratificado).
- Los síntomas con mayor poder discriminativo incluyen: `fiebre`, `dolor abdominal agudo`, `dificultad para respirar`, `náuseas`, `tos`.

### Mapeo entre datasets

Se construyó un diccionario de **297 diagnósticos sindrómicos** del hospital mapeados a síntomas del dataset público, permitiendo conectar ambas fuentes sin necesidad de re-etiquetar datos.

---

## 🤖 Modelos entrenados

### Modelo de Enfermedad

Compara 3 algoritmos sobre 721 clases con datos escalados (StandardScaler para Logistic Regression):

| Algoritmo | Requiere scaler | Naturaleza |
|---|---|---|
| Random Forest | No | Ensemble de árboles |
| Extra Trees | No | Ensemble extremadamente aleatorizado |
| **Logistic Regression** ✅ | **Sí** | **Modelo lineal multinomial** |

### Modelo de Triaje

Compara 4 algoritmos sobre datos ya escalados (5 features):

| Algoritmo | Particularidad |
|---|---|
| Random Forest | `class_weight='balanced'` |
| Extra Trees | `class_weight='balanced'` |
| **Gradient Boosting** ✅ | **Ganador — captura relaciones ordinales no lineales** |
| Logistic Regression | `class_weight='balanced'` — mal resultado (ver análisis abajo) |

#### ¿Por qué Logistic Regression falla en triaje?

LR obtuvo solo **22.6% de accuracy** (peor estadísticamente que azar) porque:
- El triaje es **ordinal** (1 < 2 < 3 < 4) y LR trata clases como independientes.
- Las 5 features (DxSindrómico codificado, sexo, edad, grupo etario, unidad) no tienen una relación **lineal** con el nivel de urgencia.
- Gradient Boosting captura estas interacciones complejas y gana consistentemente.

---

## 📈 Resultados de precisión

### 🧬 Modelo de Enfermedad — Ganador: Logistic Regression

| Modelo | Accuracy | F1 Ponderado | F1 Macro | Top-3 Accuracy |
|---|---|---|---|---|
| Random Forest | 70.28% | 75.21% | 66.74% | 81.57% |
| Extra Trees | 68.75% | 74.21% | 66.06% | 79.29% |
| **Logistic Regression** | **84.33%** | **84.77%** | **79.77%** | **95.63%** |

> 🏆 **Top-3 Accuracy de 95.63%** significa que en el 95.6% de los casos la enfermedad real está entre las 3 predicciones presentadas — métrica clínicamente más relevante con 721 clases.

### 🚨 Modelo de Triaje — Ganador: Gradient Boosting

| Modelo | Accuracy | F1 Ponderado | F1 Macro | Top-3 Accuracy |
|---|---|---|---|---|
| Random Forest | 59.05% | 61.20% | 47.08% | 99.89% |
| Extra Trees | 56.03% | 58.43% | 45.37% | 99.81% |
| **Gradient Boosting** | **62.67%** | **64.64%** | **49.28%** | **99.92%** |
| Logistic Regression | 22.58% | 18.56% | 18.04% | 93.73% |

> La precisión del triaje refleja la dificultad del problema: con solo 5 variables demográficas y sin datos de signos vitales, un 62.7% supera significativamente la predicción aleatoria (25%).

---

## ⚙️ Instalación y configuración

### Requisitos

- Python 3.12+
- scikit-learn ≥ 1.5

### 1. Clonar el repositorio

```bash
git clone https://github.com/alejohouse/bootcamp-triaje-prediccion.git
cd bootcamp-triaje-prediccion
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### 3. Colocar los archivos de datos

```
insumos/
├── sintomas_vs_enfermedades.xlsx               ← NO incluido (249 MB)
└── Morbilidad_urgencias_Hospital_Pitalito_20260406.xlsx
```

### 4. Entrenar los modelos (solo si no tienes los `.pkl`)

```bash
# Preparar y balancear los datos
python src/data_preparation.py

# Entrenar modelo de enfermedad (≈ 15 min — LR sobre 702K muestras)
python src/models.py --tipo enfermedad

# Entrenar modelo de triaje (≈ 2 min)
python src/models.py --tipo triaje
```

> ✅ Si ya tienes los archivos `.pkl` en `models/`, puedes saltar directamente al paso 5.

### 5. Lanzar la aplicación

```bash
streamlit run app.py
```

La app abrirá automáticamente en `http://localhost:8501`.

---

## 🖥️ Cómo usar la aplicación

### Paso 1 — Información del paciente (barra lateral)

Ingresa en el panel izquierdo:
- **Sexo**: Masculino / Femenino
- **Edad**: el grupo etario se calcula automáticamente
- **Unidad de atención**: selecciona la unidad de urgencias correspondiente

### Paso 2 — Selección de síntomas

Despliega las categorías de síntomas (cardiovascular, neurológico, gastrointestinal, etc.) y marca todos los que presenta el paciente.

### Paso 3 — Interpretar los resultados

Una vez seleccionado al menos un síntoma, el sistema muestra:

**Panel izquierdo — Triaje:**
- Nivel de triaje 1–4 con color y descripción
- Gauge chart de prioridad de atención
- DxSindrómico asociado (diagnóstico sindrómico más similar)

**Panel derecho — Enfermedades:**
- Top 3 enfermedades con porcentaje de confianza
- Gráfico de barras de confianza relativa

### Niveles de triaje

| Nivel | Color | Descripción | Tiempo recomendado |
|---|---|---|---|
| 🔴 **1** | Rojo | Resucitación | Inmediato |
| 🟠 **2** | Naranja | Emergencia | ≤ 15 minutos |
| 🟡 **3** | Amarillo | Urgencia | ≤ 30 minutos |
| 🟢 **4** | Verde | Menos urgente | ≤ 120 minutos |

---

## 🗂️ Estructura del proyecto

```
bootcamp-triaje-prediccion/
├── insumos/                    # Datos de entrada (archivos grandes excluidos del repo)
├── models/                     # Modelos entrenados *.pkl (excluidos del repo)
│   ├── enfermedad_model.pkl    # Logistic Regression (84.33% acc)
│   ├── enfermedad_scaler.pkl   # StandardScaler para LR
│   ├── enfermedad_label_encoder.pkl
│   ├── sintomas_columns.pkl
│   ├── triaje_model.pkl        # Gradient Boosting (62.67% acc)
│   ├── triaje_scaler.pkl
│   └── label_encoders.pkl
├── outputs/                    # Gráficas y reportes de evaluación
├── notebooks/
│   ├── 01_eda_diseases.ipynb   # EDA del dataset de enfermedades
│   ├── 02_eda_morbilidad.ipynb # EDA de los registros del hospital
│   └── 03_mapeo_datasets.ipynb # Análisis del mapeo entre datasets
├── src/
│   ├── config.py               # Rutas, parámetros y diccionario de mapeo
│   ├── data_preparation.py     # Carga, limpieza y balanceo de datos
│   ├── models.py               # Entrenamiento (--tipo enfermedad / triaje)
│   ├── pipeline.py             # Pipeline unificado de predicción
│   ├── evaluation.py           # Evaluación contra datos reales del hospital
│   └── mapeo_nlp.py            # Motor NLP TF-IDF + RapidFuzz para mapeo
├── app.py                      # Dashboard Streamlit
├── requirements.txt
├── documentacion_proyecto.md   # Documentación técnica del código
└── README.md                   # Este archivo
```

---

## ⚠️ Aviso legal

> Este sistema es una **herramienta de apoyo académica** desarrollada con datos del Hospital de Pitalito.  
> Las predicciones son orientativas y **no sustituyen el diagnóstico de un profesional de la salud**.  
> Siempre consulte con un médico calificado ante cualquier situación de urgencia.

---

<div align="center">

Desarrollado con ❤️ — Bootcamp de Ciencia de Datos 2026

</div>
