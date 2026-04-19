"""
🏥 Dashboard de Predicción Médica — Streamlit
Interfaz interactiva para predecir enfermedad y nivel de triaje.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    SINDROMATICO_A_SINTOMAS,
    TRIAJE_LABELS, MODELS_DIR
)
from src.pipeline import MedicalPredictionPipeline

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="🏥 Predictor Médico — Triaje & Enfermedad",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS PERSONALIZADO
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #00d2ff, #3a7bd5, #00d2ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: shine 3s linear infinite;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
    }

    .triage-card {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }

    .triage-1 { background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; }
    .triage-2 { background: linear-gradient(135deg, #e67e22, #d35400); color: white; }
    .triage-3 { background: linear-gradient(135deg, #f1c40f, #f39c12); color: #2c3e50; }
    .triage-4 { background: linear-gradient(135deg, #2ecc71, #27ae60); color: white; }

    .disease-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }

    .metric-container {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .sidebar .sidebar-content {
        background: rgba(15, 12, 41, 0.95);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    .stSelectbox label, .stMultiSelect label, .stNumberInput label, .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CARGAR PIPELINE
# ============================================================
@st.cache_resource
def load_pipeline():
    """Carga el pipeline de predicción (cacheado)."""
    return MedicalPredictionPipeline()


def get_triaje_color(level):
    """Retorna el color CSS para el nivel de triaje."""
    colors = {
        1: ('#e74c3c', '🔴'),
        2: ('#e67e22', '🟠'),
        3: ('#f1c40f', '🟡'),
        4: ('#2ecc71', '🟢'),
    }
    return colors.get(level, ('#95a5a6', '⚪'))


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================
def main():
    # Header
    st.markdown('<h1 class="hero-title">🏥 Predictor Médico Inteligente</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Sistema de predicción de enfermedad y nivel de triaje basado en Machine Learning</p>', unsafe_allow_html=True)

    # Verificar que los modelos existen
    modelos_requeridos = ['enfermedad_model.pkl', 'triaje_model.pkl']
    modelos_faltantes = [m for m in modelos_requeridos
                         if not os.path.exists(os.path.join(MODELS_DIR, m))]
    if modelos_faltantes:
        st.error("⚠️ Los modelos no han sido entrenados. Ejecuta primero:")
        st.code(
            "python src/data_preparation.py\n"
            "python src/models.py --tipo enfermedad\n"
            "python src/models.py --tipo triaje"
        )
        st.info(f"Modelos faltantes: {', '.join(modelos_faltantes)}")
        return

    pipeline = load_pipeline()

    # ============================================================
    # SIDEBAR: Input del paciente
    # ============================================================
    with st.sidebar:
        st.markdown("## 👤 Información del Paciente")
        st.markdown("---")

        sexo = st.radio("**Sexo**", ["Masculino", "Femenino"], horizontal=True)

        edad = st.number_input("**Edad (años)**", min_value=0, max_value=120, value=30, step=1)

        # Determinar grupo etario automáticamente
        if edad < 1:
            grupo_etario = "Menores de 1"
        elif edad <= 4:
            grupo_etario = "Entre 1 y 4"
        elif edad <= 14:
            grupo_etario = "Entre 5 y 14"
        elif edad <= 44:
            grupo_etario = "Entre 15 y 44"
        elif edad <= 59:
            grupo_etario = "Entre 45 y 59"
        else:
            grupo_etario = "Mayores de 60"
        st.info(f"📋 Grupo etario: **{grupo_etario}**")

        unidad = st.selectbox(
            "**Unidad de Atención**",
            ["URGENCIAS CONSULTA Y PROCEDIMIENTOS",
             "URGENCIAS TRAUMA Y PROCEDIMIENTOS",
             "URGENCIAS OBSERVACION REANIMACION"],
        )

        st.markdown("---")
        st.markdown("## 🩺 Síntomas del Paciente")

        # Síntomas organizados por categoría (en español)
        symptom_categories = {
            "🫀 Cardiovascular / Respiratorio": [
                "dolor agudo en el pecho", "opresión en el pecho", "dificultad para respirar",
                "palpitaciones", "arritmia", "respirando rápido", "toser expectorando",
            ],
            "🧠 Neurológico": [
                "dolor de cabeza", "mareo", "desmayo", "convulsiones",
                "movimientos involuntarios anormales", "insomnio",
            ],
            "🤢 Gastrointestinal": [
                "dolor abdominal agudo", "náuseas", "vómitos", "diarrea",
                "distensión abdominal", "flatulencia", "sangre en las heces",
            ],
            "🤒 Sistémico": [
                "fiebre", "escalofríos", "dolor por todas partes",
                "ganglios linfáticos inflamados", "sentirse mal",
            ],
            "👃 Oído / Nariz / Garganta": [
                "dolor de garganta", "amígdalas inflamadas o enrojecidas", "congestión sinusal",
                "voz ronca", "dolor de oído", "zumbido en el oído", "hemorragia nasal",
            ],
            "🦴 Musculoesquelético": [
                "lumbalgia", "dolor de espalda", "dolor en la pierna", "dolor de cadera",
                "dolor en el brazo", "hinchazón de espalda",
            ],
            "🔬 Urológico / Genital": [
                "micción dolorosa", "micción frecuente", "dolor suprapúbico",
                "retención de orina",
            ],
            "🧠 Psicológico": [
                "ansiedad y nerviosismo", "depresión",
                "síntomas depresivos o psicóticos", "comportamiento hostil",
            ],
            "🩹 Piel / Heridas": [
                "lesión cutánea", "erupción cutánea", "descarga",
            ],
        }

        sintomas_seleccionados = []
        for category, symptoms in symptom_categories.items():
            with st.expander(category):
                for symptom in symptoms:
                    if st.checkbox(symptom, key=f"symptom_{symptom}"):
                        sintomas_seleccionados.append(symptom)

        st.markdown("---")
        n_selected = len(sintomas_seleccionados)
        if n_selected > 0:
            st.success(f"✅ {n_selected} síntomas seleccionados")
        else:
            st.warning("⚠️ Selecciona al menos un síntoma")

    # ============================================================
    # CONTENIDO PRINCIPAL
    # ============================================================
    if len(sintomas_seleccionados) == 0:
        # Pantalla de inicio
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-container">
                <h3>🧬 773</h3>
                <p>Enfermedades en el modelo</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-container">
                <h3>🩺 382</h3>
                <p>Síntomas evaluados</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-container">
                <h3>🏥 59,001</h3>
                <p>Registros de entrenamiento</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📖 ¿Cómo usar este sistema?")
        st.markdown("""
        1. **Ingresa la información del paciente** en la barra lateral (sexo, edad, unidad)
        2. **Selecciona los síntomas** que presenta el paciente
        3. **El sistema predecirá automáticamente**:
           - Las **3 enfermedades más probables** con porcentaje de confianza
           - El **nivel de triaje recomendado** (1-4)
           - Una **comparación** con diagnósticos reales del Hospital de Pitalito

        > ⚠️ **Disclaimer**: Este sistema es una herramienta de apoyo educativa. No reemplaza
        > el juicio clínico de un profesional médico.
        """)
        return

    # ============================================================
    # PREDICCIÓN
    # ============================================================
    with st.spinner("🔄 Analizando síntomas..."):
        result = pipeline.predict_full(
            sintomas_list=sintomas_seleccionados,
            sexo=sexo,
            edad=edad,
            grupo_etario=grupo_etario,
            unidad=unidad
        )

    # --- Resultados ---
    st.markdown("---")

    col_triaje, col_enfermedad = st.columns([1, 2])

    # --- TRIAJE ---
    with col_triaje:
        st.markdown("### 🚨 Nivel de Triaje")
        triaje_level = result['triaje']['triaje']
        triaje_desc = result['triaje']['descripción']
        color, emoji = get_triaje_color(triaje_level)

        st.markdown(f"""
        <div class="triage-card triage-{triaje_level}">
            <div style="font-size: 3rem;">{emoji}</div>
            <div style="font-size: 2.5rem; font-weight: 700;">Nivel {triaje_level}</div>
            <div style="font-size: 1.2rem; margin-top: 0.5rem;">{triaje_desc}</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart para triaje
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=triaje_level,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Prioridad de Atención", 'font': {'color': 'white'}},
            number={'font': {'color': 'white'}},
            gauge={
                'axis': {'range': [1, 4], 'tickwidth': 1, 'tickcolor': "white",
                        'tickfont': {'color': 'white'}},
                'bar': {'color': color},
                'bgcolor': 'rgba(0,0,0,0)',
                'steps': [
                    {'range': [1, 1.5], 'color': 'rgba(231,76,60,0.3)'},
                    {'range': [1.5, 2.5], 'color': 'rgba(230,126,34,0.3)'},
                    {'range': [2.5, 3.5], 'color': 'rgba(241,196,15,0.3)'},
                    {'range': [3.5, 4], 'color': 'rgba(46,204,113,0.3)'},
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=250,
            margin=dict(t=50, b=0, l=30, r=30),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        **DxSindromatico asociado:**
        `{result['dx_sindromatico']}`
        """)

    # --- ENFERMEDADES PREDICHAS ---
    with col_enfermedad:
        st.markdown("### 🧬 Enfermedades Probables (Top 3)")

        predictions = result['enfermedad_predictions']

        for i, pred in enumerate(predictions):
            confidence_pct = pred['confidence'] * 100
            bar_color = ['#3a7bd5', '#00d2ff', '#7c4dff'][i]

            st.markdown(f"""
            <div class="disease-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: {bar_color}; font-size: 1.5rem; font-weight: 700;">#{i+1}</span>
                        <span style="color: #e2e8f0; font-size: 1.1rem; font-weight: 600; margin-left: 0.5rem;">
                            {pred['enfermedad'].title()}
                        </span>
                    </div>
                    <span style="color: {bar_color}; font-size: 1.3rem; font-weight: 700;">
                        {confidence_pct:.1f}%
                    </span>
                </div>
                <div style="margin-top: 0.5rem;">
                    <div style="background: rgba(255,255,255,0.1); border-radius: 8px; height: 8px; overflow: hidden;">
                        <div style="background: {bar_color}; width: {confidence_pct}%; height: 100%; border-radius: 8px;
                                    transition: width 0.5s ease;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Gráfico de barras de confianza
        fig_conf = go.Figure(data=[
            go.Bar(
                x=[p['confidence'] * 100 for p in predictions],
                y=[p['enfermedad'].title() for p in predictions],
                orientation='h',
                marker_color=['#3a7bd5', '#00d2ff', '#7c4dff'][:len(predictions)],
                text=[f"{p['confidence']*100:.1f}%" for p in predictions],
                textposition='outside',
                textfont=dict(color='white'),
            )
        ])
        fig_conf.update_layout(
            title=dict(text="Confianza de Predicción", font=dict(color='white')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Confianza (%)", color='white', gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(color='white'),
            height=250,
            margin=dict(t=50, b=30, l=10, r=50),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    # --- SÍNTOMAS SELECCIONADOS ---
    st.markdown("---")
    st.markdown("### 📋 Resumen del Análisis")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**Síntomas reportados:**")
        for s in sintomas_seleccionados:
            st.markdown(f"- ✅ {s}")

    with col_s2:
        st.markdown("**Datos del paciente:**")
        st.markdown(f"- 👤 Sexo: **{sexo}**")
        st.markdown(f"- 📅 Edad: **{edad} años**")
        st.markdown(f"- 📊 Grupo etario: **{grupo_etario}**")
        st.markdown(f"- 🏥 Unidad: **{unidad}**")

    # --- DISCLAIMER ---
    st.markdown("---")
    st.warning("""
    ⚠️ **Aviso importante**: Este sistema es una herramienta de apoyo basada en datos del Hospital
    de Pitalito. Las predicciones son orientativas y **no sustituyen el diagnóstico de un profesional
    de la salud**. Siempre consulte con un médico calificado.
    """)


if __name__ == "__main__":
    main()
