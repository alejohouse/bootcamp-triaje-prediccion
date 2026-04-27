#!/bin/bash
# ============================================================
#  🏥  Predictor Médico — Iniciador de aplicación (macOS)
#  Doble clic en este archivo para abrir la app en el navegador
# ============================================================

# Colores para los mensajes en terminal
AZUL='\033[1;34m'
VERDE='\033[1;32m'
ROJO='\033[1;31m'
AMARILLO='\033[1;33m'
RESET='\033[0m'

# Ir al directorio del proyecto (donde está este script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

clear
echo ""
echo -e "${AZUL}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${AZUL}║   🏥  Predictor Médico — Triaje & Enfermedad     ║${RESET}"
echo -e "${AZUL}║          Hospital de Pitalito · ML System         ║${RESET}"
echo -e "${AZUL}╚══════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Buscar entorno virtual ──────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    echo -e "${VERDE}✅  Activando entorno virtual (.venv)...${RESET}"
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo -e "${VERDE}✅  Activando entorno virtual (venv)...${RESET}"
    source venv/bin/activate
else
    echo -e "${ROJO}❌  No se encontró un entorno virtual (.venv o venv).${RESET}"
    echo -e "${AMARILLO}   Asegúrate de haberlo creado con:${RESET}"
    echo -e "   python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    echo ""
    read -p "Presiona Enter para cerrar..."
    exit 1
fi

# ── Verificar que Streamlit esté instalado ──────────────────
if ! command -v streamlit &> /dev/null; then
    echo -e "${ROJO}❌  Streamlit no está instalado en el entorno.${RESET}"
    echo -e "${AMARILLO}   Instálalo con:  pip install streamlit${RESET}"
    echo ""
    read -p "Presiona Enter para cerrar..."
    exit 1
fi

# ── Verificar que existen los modelos entrenados ────────────
MODELOS_DIR="models"
FALTANTES=()
for modelo in "enfermedad_model.pkl" "triaje_model.pkl"; do
    if [ ! -f "$MODELOS_DIR/$modelo" ]; then
        FALTANTES+=("$modelo")
    fi
done

if [ ${#FALTANTES[@]} -gt 0 ]; then
    echo -e "${AMARILLO}⚠️   Modelos no encontrados: ${FALTANTES[*]}${RESET}"
    echo -e "${AMARILLO}   La app abrirá pero pedirá entrenar los modelos primero.${RESET}"
    echo ""
fi

# ── Lanzar la aplicación ────────────────────────────────────
echo -e "${VERDE}🚀  Iniciando Predictor Médico...${RESET}"
echo -e "    URL local: ${AZUL}http://localhost:8501${RESET}"
echo ""
echo -e "${AMARILLO}   (Para cerrar la app, cierra esta ventana o presiona Ctrl+C)${RESET}"
echo ""

# Abrir el navegador después de 2 segundos (espera a que Streamlit arranque)
(sleep 2 && open "http://localhost:8501") &

# Ejecutar Streamlit
streamlit run app.py \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false

# Si Streamlit se cierra, mostrar mensaje
echo ""
echo -e "${AZUL}La aplicación se ha cerrado.${RESET}"
read -p "Presiona Enter para cerrar esta ventana..."
