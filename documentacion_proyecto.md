# 📚 Documentación del Proyecto: Modelo Predictivo de Triaje y Diagnóstico

## Descripción general

Este proyecto entrena modelos de Machine Learning que, dados los **síntomas de un paciente**, predicen:
1. **Las 3 enfermedades más probables** con porcentaje de confianza.
2. **El nivel de triaje recomendado** (1 = Resucitación → 4 = Menos urgente).

Todo funciona sobre dos fuentes de datos:
- `sintomas_vs_enfermedades.xlsx` — dataset de síntomas binarios vs. enfermedades (todo en español)
- `Morbilidad_urgencias_Hospital_Pitalito_20260406.xlsx` — registros reales de urgencias del Hospital de Pitalito

---

## 🗂️ Estructura de archivos

```
proyecto_bootcamp/
├── insumos/                        # Datos de entrada
│   ├── sintomas_vs_enfermedades.xlsx
│   └── Morbilidad_urgencias_Hospital_Pitalito_20260406.xlsx
├── models/                         # Modelos entrenados (*.pkl)
│   ├── enfermedad_model.pkl        # Mejor modelo de enfermedad
│   ├── enfermedad_label_encoder.pkl# Codificador de nombres de enfermedades
│   ├── enfermedad_scaler.pkl       # Scaler (solo si ganó Logistic Regression)
│   ├── enfermedad_model_name.txt   # Nombre del algoritmo ganador
│   ├── sintomas_columns.pkl        # Lista de síntomas válidos (columnas útiles)
│   ├── sintoma_columns.pkl         # Alias generado por data_preparation.py
│   ├── triaje_model.pkl            # Mejor modelo de triaje
│   ├── triaje_scaler.pkl           # Scaler para el modelo de triaje
│   ├── triaje_model_name.txt       # Nombre del algoritmo ganador
│   └── label_encoders.pkl          # Encoders de columnas categóricas del hospital
├── outputs/                        # Resultados exportados de scripts standalone
│   ├── enfermedad_model_comparison.png
│   ├── enfermedad_feature_importance.png
│   ├── enfermedad_classification_report.csv
│   ├── triaje_model_comparison.png
│   ├── triaje_confusion_matrix.png
│   └── triaje_classification_report.csv
├── notebooks/                      # Análisis exploratorio (resultados quedan en el notebook)
│   ├── 01_eda_diseases.ipynb
│   ├── 02_eda_morbilidad.ipynb
│   └── 03_mapeo_datasets.ipynb
├── src/                            # Código fuente del pipeline
│   ├── config.py
│   ├── data_preparation.py
│   ├── models.py                   # Script unificado de entrenamiento (--tipo enfermedad / triaje)
│   ├── pipeline.py
│   ├── evaluation.py
│   └── mapeo_nlp.py
├── app.py                          # Dashboard Streamlit
└── documentacion_proyecto.md      # Este archivo
```

---

## 🔧 Scripts del directorio `src/`

### 1. `config.py` — Configuración central

**Propósito:** Único punto de verdad para todas las rutas, parámetros y diccionarios del proyecto.

```python
import os

# BASE_DIR apunta a la carpeta raíz del proyecto.
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSUMOS_DIR  = os.path.join(BASE_DIR, "insumos")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR  = os.path.join(BASE_DIR, "outputs")

# Rutas a los archivos de datos
ENFERMEDAD_XLSX  = os.path.join(INSUMOS_DIR, "sintomas_vs_enfermedades.xlsx")
MORBILIDAD_XLSX  = os.path.join(INSUMOS_DIR, "Morbilidad_urgencias_Hospital_Pitalito_20260406.xlsx")

# Constantes de entrenamiento
RANDOM_STATE = 42    # Semilla para reproducibilidad
TEST_SIZE    = 0.2   # 20% de datos para test
CV_FOLDS     = 5

# Niveles de triaje excluidos por muy pocas muestras
TRIAJE_EXCLUIDO = [0, 5]
TRIAJE_LABELS   = {1: "Resucitación", 2: "Emergencia", 3: "Urgencia", 4: "Menos urgente"}

# SINDROMATICO_A_SINTOMAS: Puente entre los dos datasets.
# Mapea cada DxSindrómico del hospital a síntomas del xlsx (en español).
SINDROMATICO_A_SINTOMAS = { ... }  # 297 diagnósticos mapeados

# RANDOM FOREST
RF_PARAMS = {
    "n_estimators": 200, "max_depth": 20, "min_samples_split": 5,
    "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1,
}

# REGRESIÓN LOGÍSTICA
# Nota: multi_class fue eliminado en scikit-learn 1.5+ (lbfgs usa multinomial por defecto).
# Nota: n_jobs fue eliminado de LogisticRegression en scikit-learn 1.8+.
LR_PARAMS = {
    "max_iter": 1000,
    "C": 1.0,
    "solver": "lbfgs",
    "random_state": 42,
}
```

---

### 2. `data_preparation.py` — Preparación de datos

**Propósito:** Carga, limpia y divide los datos para entrenar los modelos.

#### `load_enfermedad_dataset()`

```python
def load_enfermedad_dataset():
    df = pd.read_excel(ENFERMEDAD_XLSX)

    # Normaliza nombres de enfermedades para evitar duplicados por capitalización.
    df['enfermedad'] = df['enfermedad'].str.strip().str.lower()

    sintoma_cols = df.columns[1:].tolist()

    # Filtra síntomas con varianza < 0.005 (no discriminan entre enfermedades).
    varianza      = df[sintoma_cols].var()
    var_baja_cols = varianza[varianza < 0.005].index.tolist()
    var_alta_cols = [c for c in sintoma_cols if c not in var_baja_cols]

    return df, var_alta_cols, var_baja_cols
```

#### `prepare_enfermedad_data(df, sintoma_cols)`

```python
def prepare_enfermedad_data(df, sintoma_cols):
    # Filtra enfermedades con < 5 muestras.
    enfermedad_counts  = df['enfermedad'].value_counts()
    valid_enfermedades = enfermedad_counts[enfermedad_counts >= 5].index
    df_filtered        = df[df['enfermedad'].isin(valid_enfermedades)].copy()

    X = df_filtered[sintoma_cols].values   # Matriz de 0s y 1s
    y = df_filtered['enfermedad'].values   # Array de nombres en español

    # LabelEncoder: convierte strings a enteros para sklearn.
    le_enfermedad = LabelEncoder()
    y_encoded     = le_enfermedad.fit_transform(y)

    # Split estratificado 80/20.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # SMOTE genera muestras sintéticas para clases minoritarias.
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    joblib.dump(le_enfermedad, os.path.join(MODELS_DIR, 'enfermedad_label_encoder.pkl'))
    return X_train_bal, X_test, y_train_bal, y_test, le_enfermedad
```

#### `prepare_triaje_data(df_encoded)`

```python
def prepare_triaje_data(df_encoded):
    feature_cols = [
        'DxSindromatico_encoded', 'Sexo_encoded', 'Edad',
        'GrupoEtario1_encoded', 'Unidad_encoded'
    ]

    # SMOTETomek = SMOTE + Tomek Links en triaje
    smote_tomek = SMOTETomek(random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train)

    # StandardScaler: media=0, std=1 (necesario para LR y mejora GB)
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled  = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODELS_DIR, 'triaje_scaler.pkl'))
    return X_train_scaled, X_test_scaled, y_train_bal, y_test, scaler
```

---

### 3. `models.py` — Entrenamiento unificado

**Propósito:** Script único de entrenamiento. Recibe `--tipo enfermedad` o `--tipo triaje`.

#### Modelo de Enfermedad

Compara tres algoritmos. El scaler solo se guarda si gana Logistic Regression (los árboles no lo necesitan).

```python
modelos_config = {
    "Random Forest":       (RandomForestClassifier(**RF_PARAMS),    X_train,    X_test),
    "Extra Trees":         (ExtraTreesClassifier(...),               X_train,    X_test),
    "Logistic Regression": (LogisticRegression(**LR_PARAMS),         X_train_sc, X_test_sc),
}
# Ganador → enfermedad_model.pkl + (opcional) enfermedad_scaler.pkl
```

**Resultados obtenidos:**

| Modelo | Accuracy | F1 Ponderado | Top-3 Acc |
|---|---|---|---|
| Random Forest | 0.7028 | 0.7521 | 0.8157 |
| Extra Trees | 0.6875 | 0.7421 | 0.7929 |
| **Logistic Regression** ✅ | **0.8433** | **0.8477** | **0.9563** |

Logistic Regression ganó con 84.3% de accuracy y 95.6% de Top-3 accuracy sobre 721 enfermedades.

#### Modelo de Triaje

Compara cuatro algoritmos. Todos reciben los datos ya escalados por `prepare_triaje_data`.

```python
modelos_config = {
    "Random Forest":      RandomForestClassifier(class_weight='balanced', ...),
    "Extra Trees":        ExtraTreesClassifier(class_weight='balanced', ...),
    "Gradient Boosting":  GradientBoostingClassifier(...),
    "Logistic Regression": LogisticRegression(**LR_PARAMS, class_weight='balanced'),
}
# Ganador → triaje_model.pkl
```

**Resultados obtenidos:**

| Modelo | Accuracy | F1 Ponderado | Top-3 Acc |
|---|---|---|---|
| Random Forest | 0.5905 | 0.6120 | 0.9989 |
| Extra Trees | 0.5603 | 0.5843 | 0.9981 |
| **Gradient Boosting** ✅ | **0.6267** | **0.6464** | **0.9992** |
| Logistic Regression | 0.2258 | 0.1856 | 0.9373 |

---

### 4. `pipeline.py` — Pipeline de predicción unificado

**Propósito:** Clase que integra los modelos entrenados y expone métodos de predicción para la app Streamlit.

```python
class MedicalPredictionPipeline:

    def __init__(self):
        # Carga todos los artefactos desde disco.
        self.enfermedad_model   = joblib.load('enfermedad_model.pkl')
        self.enfermedad_encoder = joblib.load('enfermedad_label_encoder.pkl')
        self.sintomas_columns   = joblib.load('sintomas_columns.pkl')

        # enfermedad_scaler existe SOLO si ganó Logistic Regression
        scaler_path = os.path.join(MODELS_DIR, 'enfermedad_scaler.pkl')
        self.enfermedad_scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        self.triaje_model    = joblib.load('triaje_model.pkl')
        self.triaje_scaler   = joblib.load('triaje_scaler.pkl')
        self.label_encoders  = joblib.load('label_encoders.pkl')

    def predict_enfermedad(self, sintomas_list):
        # 1. Construye vector binario: posición=síntoma, valor=0 o 1
        vector = np.zeros(len(self.sintomas_columns))
        for sintoma in sintomas_list:
            if sintoma in self.sintomas_columns:
                vector[self.sintomas_columns.index(sintoma)] = 1

        # 2. Escala si el modelo necesita (solo LR)
        X = vector.reshape(1, -1)
        if self.enfermedad_scaler:
            X = self.enfermedad_scaler.transform(X)

        # 3. Top-3 probabilidades → nombres de enfermedades en español
        probas      = self.enfermedad_model.predict_proba(X)[0]
        top_indices = np.argsort(probas)[-3:][::-1]
        return [{'enfermedad': self.enfermedad_encoder.inverse_transform([i])[0],
                 'confidence': float(probas[i])}
                for i in top_indices]

    def predict_triaje(self, dx_sindromatico, sexo, edad, grupo_etario, unidad):
        # Codifica variables categóricas con los encoders del hospital.
        # ValueError si el valor no fue visto en entrenamiento → retorna triaje 3 (default).
        X        = np.array([[dx_enc, sexo_enc, edad, grupo_enc, unidad_enc]])
        X_scaled = self.triaje_scaler.transform(X)
        triaje_pred = self.triaje_model.predict(X_scaled)[0]
        return {
            'triaje':      int(triaje_pred),
            'descripción': TRIAJE_LABELS.get(int(triaje_pred), f'Nivel {triaje_pred}')
        }

    def _find_closest_dx(self, sintomas_list):
        # Jaccard: |A ∩ B| / |A ∪ B| donde A=síntomas usuario, B=síntomas del DxSindrómico
        for dx, dx_sintomas in SINDROMATICO_A_SINTOMAS.items():
            score = len(set(sintomas_list) & set(dx_sintomas)) / \
                    len(set(sintomas_list) | set(dx_sintomas))
```

### 5. `evaluation.py` — Evaluación contra datos reales

**Propósito:** Mide la calidad del pipeline comparando predicciones con registros reales del Hospital de Pitalito.

```python
def evaluate_disease_on_hospital_data(pipeline, df_urgencias, sample_size=500):
    # Para cada registro del hospital con DxSindrómico conocido:
    # 1. DxSindrómico → lista de síntomas en español (via SINDROMATICO_A_SINTOMAS)
    # 2. Síntomas → predict_enfermedad() → top-3 enfermedades
    # 3. Comparación flexible: pred['enfermedad'] in real_dx OR real_dx in pred['enfermedad']
    match = any(
        pred['enfermedad'].lower() in real_dx or real_dx in pred['enfermedad'].lower()
        for pred in predictions
    )

def evaluate_triage_accuracy(pipeline, df_urgencias, sample_size=1000):
    diff = np.abs(np.array(y_true) - np.array(y_pred))
    # - Accuracy exacta: predicción == real
    # - Error Medio Absoluto: cuántos niveles se equivoca en promedio
    # - % error ≤ 1 nivel: clínicamente aceptable (urgencia vs emergencia es un nivel)
```

---

## 🖥️ `app.py` — Dashboard Streamlit

**Propósito:** Interfaz web interactiva para uso clínico en tiempo real.

```python
@st.cache_resource      # Carga el pipeline UNA SOLA VEZ por sesión (no en cada click)
def load_pipeline():
    return MedicalPredictionPipeline()

# Verificación de modelos: comprueba enfermedad_model.pkl y triaje_model.pkl
# (no disease_model.pkl — nombre anterior al refactor).
modelos_requeridos = ['enfermedad_model.pkl', 'triaje_model.pkl']

# Sidebar: Parámetros del paciente
# - Sexo, Edad (con cálculo automático del grupo etario con if/elif)
# - Unidad de atención (3 opciones del hospital)
# - Síntomas: checkboxes en español, organizados por categoría corporal

# Predicción en tiempo real al seleccionar un síntoma:
result = pipeline.predict_full(sintomas_list=sintomas_seleccionados, ...)

# Visualizaciones con Plotly (interactivas):
# - Gauge chart del nivel de triaje (1=rojo → 4=verde)
# - Barras horizontales de confianza para las top-3 enfermedades
# - Tarjetas con gradientes de colores por nivel de triaje (HTML/CSS inline)
```

---

## 🧪 Notebooks de análisis exploratorio

> Los resultados permanecen en el notebook como salida inline. No se exportan imágenes a `outputs/`.

| Notebook | Contenido |
|---|---|
| `01_eda_diseases.ipynb` | Distribución de enfermedades, síntomas más frecuentes, correlaciones, análisis de varianza |
| `02_eda_morbilidad.ipynb` | Distribución de triaje, top diagnósticos, perfil demográfico, análisis temporal |
| `03_mapeo_datasets.ipynb` | Cobertura del mapeo: ¿qué % de DxSindrómicos están en SINDROMATICO_A_SINTOMAS? |

---

## 🚀 Orden de ejecución

```bash
# 1. Preparar datos (lee xlsx, aplica SMOTE, guarda encoders y scaler)
python src/data_preparation.py

# 2. Entrenar modelo de enfermedades (compara RF, ET, LR → guarda el mejor)
python src/models.py --tipo enfermedad

# 3. Entrenar modelo de triaje (compara RF, ET, GB, LR → guarda el mejor)
python src/models.py --tipo triaje

# 4. (Opcional) Evaluar contra datos reales del hospital
python src/evaluation.py

# 5. Lanzar la aplicación web
streamlit run app.py
```

> ⚠️ **No es necesario re-entrenar** si los archivos `.pkl` ya existen en `models/`.
> Streamlit carga directamente los modelos guardados.

---

## 📊 Artefactos exportados

| Destino | Archivos | Generado por |
|---|---|---|
| `models/` | `enfermedad_model.pkl`, `triaje_model.pkl`, `*_scaler.pkl`, `*_encoder.pkl`, `sintomas_columns.pkl`, `label_encoders.pkl` | `data_preparation.py`, `models.py` |
| `outputs/` | `*_comparison.png`, `triaje_confusion_matrix.png`, `*_report.csv`, `enfermedad_feature_importance.png` | `models.py` |

Los notebooks NO exportan imágenes a `outputs/` — sus visualizaciones quedan como salida inline del notebook.

---

## 🔍 Decisiones de diseño

| Decisión | Razón |
|---|---|
| SMOTE en enfermedades | 721 clases muy desbalanceadas; SMOTE balancea sin eliminar datos |
| SMOTETomek en triaje | Limpia además las fronteras de decisión entre niveles adyacentes |
| Filtro varianza < 0.005 | Elimina síntomas sin poder discriminativo (164 de 377 eliminados) |
| Umbral ≥ 5 muestras | Permite split estratificado sin errores (elimina 52 enfermedades raras) |
| Similaridad Jaccard | Conecta síntomas con DxSindrómico sin modelo adicional |
| Regresión Logística en competencia | Ganó en enfermedades (84.3% acc vs 70.3% RF); en triaje pierde por naturaleza ordinal del problema |
| Top-3 accuracy | Con 721 clases, ver las 3 opciones es más valioso que solo la 1 |
| Gradient Boosting ganó en triaje | Captura relaciones no lineales entre variables ordinales (triaje es ordinal) |

---

## ⚙️ Compatibilidad con scikit-learn

> Versión mínima requerida: **scikit-learn 1.5+** (Python 3.12)

| Parámetro eliminado | Motivo | Solución aplicada |
|---|---|---|
| `multi_class='multinomial'` en `LogisticRegression` | Eliminado en sklearn 1.5; `lbfgs` ya lo hace por defecto | Removido de `LR_PARAMS` |
| `n_jobs` en `LogisticRegression` | Sin efecto desde sklearn 1.8, será eliminado en 1.10 | Removido de `LR_PARAMS` |

---

## 📈 ¿Por qué Logistic Regression funciona bien en enfermedades pero mal en triaje?

### En enfermedades (721 clases, datos de síntomas binarios)
- El dataset de síntomas tiene **estructura clara y separable**: cada enfermedad tiene un perfil de síntomas específico.
- LR aprende **un hiperplano separador por clase** (One-vs-Rest implícito con lbfgs multinomial).
- Con `StandardScaler` aplicado, las 213 features binarias están correctamente normalizadas.
- **Resultado**: 84.3% accuracy, superando a Random Forest (70.3%).

### En triaje (4 clases, datos demográficos + DxSindrómico)
- El triaje depende de la **gravedad clínica**, que es inherentemente **ordinal** (1 < 2 < 3 < 4).
- LR asume independencia entre clases, ignorando que "Urgencia" y "Emergencia" son adyacentes.
- Con solo **5 features** (DxSindrómico, Sexo, Edad, GrupoEtario, Unidad), hay poca señal para separar 4 clases ordinales.
- El DxSindrómico codificado como entero no tiene una relación lineal con el triaje.
- **Resultado**: 22.6% accuracy — peor que adivinar al azar (25% esperado); claramente inapropiado.
- **Gradient Boosting** gana porque captura interacciones no lineales complejas entre los features.

> **Conclusión**: Logistic Regression **sí está implementada en el modelo de triaje** y participa en la competencia, pero es consistentemente descartada por el selector automático. El modelo ganador (`Gradient Boosting`) se guarda en `triaje_model.pkl`.
