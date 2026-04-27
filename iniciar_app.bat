@echo off
chcp 65001 >nul
title 🏥 Predictor Médico — Iniciando...

:: ============================================================
::  🏥  Predictor Médico — Iniciador de aplicación (Windows)
::  Doble clic en este archivo para abrir la app en el navegador
:: ============================================================

cls
echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║   🏥  Predictor Médico — Triaje ^& Enfermedad     ║
echo  ║          Hospital de Pitalito · ML System         ║
echo  ╚══════════════════════════════════════════════════╝
echo.

:: Ir al directorio donde está este archivo .bat
cd /d "%~dp0"

:: ── Buscar entorno virtual ─────────────────────────────────
if exist ".venv\Scripts\activate.bat" (
    echo  [OK] Activando entorno virtual ^(.venv^)...
    call .venv\Scripts\activate.bat
    goto :CHECK_STREAMLIT
)

if exist "venv\Scripts\activate.bat" (
    echo  [OK] Activando entorno virtual ^(venv^)...
    call venv\Scripts\activate.bat
    goto :CHECK_STREAMLIT
)

echo  [ERROR] No se encontró un entorno virtual ^(.venv o venv^).
echo.
echo  Créalo con los siguientes comandos en la terminal:
echo    python -m venv .venv
echo    .venv\Scripts\activate
echo    pip install -r requirements.txt
echo.
pause
exit /b 1

:: ── Verificar que Streamlit esté instalado ─────────────────
:CHECK_STREAMLIT
where streamlit >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Streamlit no está instalado en el entorno.
    echo  Instálalo con:  pip install streamlit
    echo.
    pause
    exit /b 1
)

:: ── Verificar modelos entrenados ───────────────────────────
set MODELOS_OK=1
if not exist "models\enfermedad_model.pkl" set MODELOS_OK=0
if not exist "models\triaje_model.pkl"    set MODELOS_OK=0

if %MODELOS_OK%==0 (
    echo  [AVISO] Algunos modelos no han sido entrenados.
    echo  La app abrirá pero pedirá entrenar los modelos primero.
    echo.
)

:: ── Lanzar la aplicación ───────────────────────────────────
echo  [OK] Iniciando Predictor Médico...
echo       URL local: http://localhost:8501
echo.
echo  ^(Para cerrar la app, cierra esta ventana o presiona Ctrl+C^)
echo.

title 🏥 Predictor Médico — Ejecutando en http://localhost:8501

:: Abrir el navegador tras 3 segundos (en paralelo)
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8501"

:: Ejecutar Streamlit
streamlit run app.py ^
    --server.port 8501 ^
    --server.headless false ^
    --browser.gatherUsageStats false

:: Si Streamlit se cierra, mostrar mensaje
echo.
echo  La aplicación se ha cerrado.
pause
