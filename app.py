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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Base ── */
    html, body, .main, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f7fafd !important;
        color: #1a2744;
    }

    /* ── Header hero ── */
    .hero-wrapper {
        background: linear-gradient(120deg, #1a6fbd 0%, #2a9df4 60%, #56c0f7 100%);
        border-radius: 20px;
        padding: 2.2rem 2rem 1.8rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 6px 30px rgba(26,111,189,0.18);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin: 0 0 0.4rem;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.88);
        text-align: center;
        margin: 0;
        font-weight: 400;
    }
    .hero-badge {
        display: flex;
        justify-content: center;
        gap: 0.8rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .hero-badge span {
        background: rgba(255,255,255,0.2);
        color: #fff;
        border-radius: 20px;
        padding: 0.25rem 0.9rem;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* ── Sidebar: forzar fondo claro en TODOS los niveles anidados ── */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div,
    div[data-testid="stSidebar"],
    div[data-testid="stSidebar"] > div,
    div[data-testid="stSidebarContent"],
    div[data-testid="stSidebarUserContent"],
    div[data-testid="stSidebarUserContent"] > div {
        background: linear-gradient(180deg, #f0f8ff 0%, #e3f0fb 100%) !important;
        background-color: #f0f8ff !important;
    }
    section[data-testid="stSidebar"] {
        border-right: 2px solid #b8d9f5 !important;
    }

    /* Texto: legible sobre fondo claro */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #1a2744 !important;
    }

    /* Inputs text / number: fondo blanco, letra oscura */
    section[data-testid="stSidebar"] input[type="text"],
    section[data-testid="stSidebar"] input[type="number"] {
        background-color: #ffffff !important;
        color: #1a2744 !important;
        border: 1px solid #a8cff0 !important;
        border-radius: 8px !important;
        caret-color: #1a6fbd !important;
    }
    section[data-testid="stSidebar"] input:focus {
        border-color: #1a6fbd !important;
        box-shadow: 0 0 0 2px rgba(26,111,189,0.18) !important;
        outline: none !important;
    }

    /* Contenedor interno de widgets BaseWeb (selectbox, number) */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="input"] > div,
    section[data-testid="stSidebar"] [data-baseweb="base-input"] {
        background-color: #ffffff !important;
        border-color: #a8cff0 !important;
        color: #1a2744 !important;
    }

    /* Expanders: cabecera azul oscuro con texto blanco para contraste */
    section[data-testid="stSidebar"] details {
        background: #ffffff !important;
        border: 1px solid #b8d9f5 !important;
        border-radius: 10px !important;
        margin-bottom: 0.4rem !important;
        overflow: hidden !important;
    }
    section[data-testid="stSidebar"] details summary {
        background: linear-gradient(90deg, #1a5fa0 0%, #2a9df4 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.87rem !important;
        padding: 0.6rem 0.9rem !important;
        cursor: pointer !important;
        letter-spacing: 0.2px !important;
    }
    section[data-testid="stSidebar"] details summary:hover {
        background: linear-gradient(90deg, #114d87 0%, #1a8de0 100%) !important;
    }
    section[data-testid="stSidebar"] details summary svg,
    section[data-testid="stSidebar"] details summary path {
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }
    section[data-testid="stSidebar"] details[open] {
        border-color: #1a6fbd !important;
    }

    /* Checkboxes dentro del expander */
    section[data-testid="stSidebar"] details label {
        color: #1a2744 !important;
        font-size: 0.86rem !important;
    }

    /* Separadores en sidebar */
    section[data-testid="stSidebar"] hr {
        border-color: #b8d9f5 !important;
    }

    /* ── Triage cards ── */
    .triage-card {
        padding: 1.5rem 1.2rem;
        border-radius: 18px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
        border: none;
    }
    .triage-1 {
        background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
        color: #fff;
    }
    .triage-2 {
        background: linear-gradient(135deg, #dd6b20 0%, #c05621 100%);
        color: #fff;
    }
    .triage-3 {
        background: linear-gradient(135deg, #d69e2e 0%, #b7791f 100%);
        color: #fff;
    }
    .triage-4 {
        background: linear-gradient(135deg, #38a169 0%, #276749 100%);
        color: #fff;
    }

    /* ── Disease cards ── */
    .disease-card {
        background: #ffffff;
        border: 1px solid #d0e8fa;
        border-left: 4px solid #2a9df4;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin: 0.55rem 0;
        box-shadow: 0 2px 10px rgba(26,111,189,0.07);
        transition: box-shadow 0.2s;
    }
    .disease-card:hover {
        box-shadow: 0 4px 18px rgba(26,111,189,0.15);
    }

    /* ── Metric cards ── */
    .metric-container {
        background: #ffffff;
        border: 1px solid #d0e8fa;
        border-radius: 16px;
        padding: 1.6rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 14px rgba(26,111,189,0.08);
    }
    .metric-container h3 {
        font-size: 2rem;
        font-weight: 800;
        color: #1a6fbd;
        margin: 0 0 0.3rem;
    }
    .metric-container p {
        color: #4a6080;
        font-size: 0.9rem;
        margin: 0;
        font-weight: 500;
    }

    /* ── Section titles ── */
    h3 { color: #1a2744 !important; font-weight: 700 !important; }

    /* ── Labels ── */
    .stSelectbox label, .stMultiSelect label, .stNumberInput label, .stRadio label,
    .stCheckbox label, .stExpander summary {
        color: #1a2744 !important;
        font-weight: 500 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(120deg, #1a6fbd, #2a9df4);
        color: #fff;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.4rem;
        box-shadow: 0 3px 12px rgba(26,111,189,0.25);
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 16px rgba(26,111,189,0.35);
    }

    /* ── Divider ── */
    hr { border-color: #d0e8fa !important; }

    /* ── Info / warning boxes ── */
    .stAlert {
        border-radius: 10px !important;
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
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-title">🏥 Predictor Médico Inteligente</div>
        <div class="hero-subtitle">Sistema de predicción de enfermedad y nivel de triaje basado en Machine Learning</div>
        <div class="hero-badge">
            <span>🧬 Machine Learning</span>
            <span>🏥 Hospital de Pitalito</span>
            <span>📊 59 001 registros</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

        nombre_paciente = st.text_input(
            "**Nombre del paciente**",
            placeholder="Ej: Juan Pérez",
            help="Nombre identificador del paciente (opcional)"
        )

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
                <h3>🏥 59 001</h3>
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
            title={'text': "Prioridad de Atención", 'font': {'color': '#1a2744', 'size': 14}},
            number={'font': {'color': '#1a2744', 'size': 32}},
            gauge={
                'axis': {'range': [1, 4], 'tickwidth': 1, 'tickcolor': '#4a6080',
                        'tickfont': {'color': '#4a6080'}},
                'bar': {'color': color},
                'bgcolor': '#f0f6ff',
                'bordercolor': '#d0e8fa',
                'steps': [
                    {'range': [1, 1.5], 'color': 'rgba(229,62,62,0.15)'},
                    {'range': [1.5, 2.5], 'color': 'rgba(221,107,32,0.15)'},
                    {'range': [2.5, 3.5], 'color': 'rgba(214,158,46,0.15)'},
                    {'range': [3.5, 4], 'color': 'rgba(56,161,105,0.15)'},
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=250,
            margin=dict(t=50, b=0, l=30, r=30),
            font=dict(family='Inter, sans-serif'),
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
            bar_color = ['#1a6fbd', '#2a9df4', '#56c0f7'][i]

            st.markdown(f"""
            <div class="disease-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: {bar_color}; font-size: 1.4rem; font-weight: 800;">#{i+1}</span>
                        <span style="color: #1a2744; font-size: 1rem; font-weight: 600; margin-left: 0.6rem;">
                            {pred['enfermedad'].title()}
                        </span>
                    </div>
                    <span style="background:{bar_color}; color:#fff; font-size:0.95rem; font-weight:700;
                                 padding:0.2rem 0.7rem; border-radius:20px;">
                        {confidence_pct:.1f}%
                    </span>
                </div>
                <div style="margin-top: 0.6rem;">
                    <div style="background: #e8f4fd; border-radius: 8px; height: 7px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, {bar_color}, #56c0f7);
                                    width: {confidence_pct}%; height: 100%; border-radius: 8px;
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
                marker_color=['#1a6fbd', '#2a9df4', '#56c0f7'][:len(predictions)],
                text=[f"{p['confidence']*100:.1f}%" for p in predictions],
                textposition='outside',
                textfont=dict(color='#1a2744'),
            )
        ])
        fig_conf.update_layout(
            title=dict(text="Confianza de Predicción", font=dict(color='#1a2744', size=14)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#f7fafd',
            xaxis=dict(title="Confianza (%)", color='#4a6080',
                       gridcolor='#d0e8fa', showgrid=True),
            yaxis=dict(color='#1a2744'),
            height=250,
            margin=dict(t=50, b=30, l=10, r=60),
            font=dict(family='Inter, sans-serif'),
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
        if nombre_paciente:
            st.markdown(f"- 🏷️ Paciente: **{nombre_paciente}**")
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
