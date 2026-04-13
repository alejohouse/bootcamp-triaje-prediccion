# 🏥 Predictor Médico Inteligente — Triaje & Enfermedad

Sistema de **Machine Learning** para predicción de enfermedades y niveles de triaje hospitalario, desarrollado con datos reales del Hospital de Pitalito (Colombia). Interfaz interactiva construida con **Streamlit**.

---

## 📋 Descripción del Proyecto

Este proyecto integra dos fuentes de datos para construir un pipeline de predicción médica end-to-end:

| Dataset | Descripción |
|---|---|
| `Final_Augmented_dataset_Diseases_and_Symptoms.csv` | +190 MB · 773 enfermedades · 382 síntomas |
| `Morbilidad_urgencias_Hospital_Pitalito_20260406.xlsx` | Registros reales de urgencias del hospital |
| `TriageUrgencias.csv` | Histórico de niveles de triaje asignados |
| `registro_procedimientos_medicos_diagnosticos_2019.csv` | Procedimientos y diagnósticos (2019) |

### Qué predice el sistema

1. **Enfermedad probable** → Top 3 diagnósticos con porcentaje de confianza
2. **Nivel de triaje** → Escala 1–4 (Resucitación → Menos urgente)

---

## 🗂️ Estructura del Proyecto

```
proyecto_bootcamp/
│
├── app.py                      # Dashboard Streamlit (punto de entrada)
├── requirements.txt            # Dependencias Python
│
├── src/                        # Módulos del pipeline ML
│   ├── config.py               # Rutas, constantes y diccionarios de mapeo
│   ├── data_preparation.py     # Limpieza y preparación de datos
│   ├── model_disease.py        # Entrenamiento del modelo de enfermedades (Random Forest)
│   ├── model_triage.py         # Entrenamiento del modelo de triaje (Gradient Boosting)
│   ├── pipeline.py             # Pipeline unificado de predicción
│   └── evaluation.py           # Evaluación y métricas de los modelos
│
├── insumos/                    # Datos fuente (no subir a Git — ver .gitignore)
│   ├── Final_Augmented_dataset_Diseases_and_Symptoms.csv
│   ├── Morbilidad_urgencias_Hospital_Pitalito_20260406.xlsx
│   ├── TriageUrgencias.csv
│   └── registro_procedimientos_medicos_diagnosticos_2019.csv
│
├── models/                     # Modelos entrenados (.pkl) — generados automáticamente
│   ├── disease_model.pkl       # Modelo Random Forest para enfermedades
│   ├── triage_model.pkl        # Modelo Gradient Boosting para triaje
│   ├── disease_label_encoder.pkl
│   ├── triage_scaler.pkl
│   ├── symptom_columns.pkl
│   └── label_encoders.pkl
│
├── notebooks/                  # Análisis exploratorio (EDA)
│   ├── 01_eda_diseases.ipynb
│   ├── 02_eda_morbilidad.ipynb
│   └── 03_mapeo_datasets.ipynb
│
└── outputs/                    # Métricas, reportes y visualizaciones generadas
```

---

## ⚙️ Instalación

### Requisitos previos

- Python **3.10+**
- Git
- (Recomendado) un entorno virtual

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd proyecto_bootcamp
```

### 2. Crear y activar un entorno virtual

```bash
# Crear el entorno
python -m venv .venv

# Activar en macOS / Linux
source .venv/bin/activate

# Activar en Windows
.venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🤖 Entrenamiento de los Modelos

> Sáltate este paso si la carpeta `models/` ya contiene los archivos `.pkl`.

Ejecuta los scripts en el siguiente orden:

```bash
# 1. Preparar y limpiar los datos
python src/data_preparation.py

# 2. Entrenar el modelo de predicción de enfermedades (Random Forest)
python src/model_disease.py

# 3. Entrenar el modelo de triaje (Gradient Boosting)
python src/model_triage.py

# (Opcional) Evaluar el rendimiento de los modelos
python src/evaluation.py
```

> ⚠️ El entrenamiento del modelo de enfermedades puede tardar varios minutos dada la dimensión del dataset (~190 MB).

---

## 🚀 Lanzar la Aplicación en Streamlit

Con el entorno virtual activo y los modelos entrenados, ejecuta:

```bash
streamlit run app.py
```

Streamlit abrirá automáticamente el navegador en:

```
http://localhost:8501
```

### Opciones útiles al lanzar

```bash
# Especificar un puerto diferente
streamlit run app.py --server.port 8502

# Deshabilitar la apertura automática del navegador
streamlit run app.py --server.headless true

# Permitir conexiones desde otros equipos en la red
streamlit run app.py --server.address 0.0.0.0
```

---

## 🧠 Modelos de Machine Learning

| Modelo | Algoritmo | Tarea |
|---|---|---|
| **Enfermedades** | Random Forest | Clasificación multiclase (773 clases) |
| **Triaje** | Gradient Boosting | Clasificación ordinal (niveles 1–4) |

### Niveles de Triaje

| Nivel | Descripción | Color |
|---|---|---|
| 1 | 🔴 Resucitación | Atención inmediata |
| 2 | 🟠 Emergencia | Atención en < 15 min |
| 3 | 🟡 Urgencia | Atención en < 30 min |
| 4 | 🟢 Menos urgente | Puede esperar |

---

## 📊 Datos de Entrenamiento

| Métrica | Valor |
|---|---|
| Registros de entrenamiento | ~59,001 |
| Enfermedades clasificadas | 773 |
| Síntomas evaluados | 382 |
| División train/test | 80% / 20% |
| Validación cruzada | 5 folds |

---

## 📦 Dependencias Principales

```txt
streamlit>=1.30
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
plotly>=5.18
imbalanced-learn>=0.11
joblib>=1.3
openpyxl>=3.1
```

---

## ⚠️ Disclaimer

> Este sistema es una **herramienta de apoyo educativa** desarrollada como proyecto de bootcamp.  
> Las predicciones son orientativas y **no sustituyen el diagnóstico de un profesional de la salud**.  
> Siempre consulte con un médico calificado ante cualquier síntoma.

---

## 👥 Equipo

Proyecto desarrollado como trabajo de grado en el marco del **Bootcamp de Ciencia de Datos**.  
Datos proporcionados por el **Hospital de Pitalito, Huila — Colombia**.

---

*Última actualización: Abril 2026*
