@echo off
REM Ir a la carpeta donde está este .bat
cd /d "%~dp0"

echo Activando entorno virtual...

REM Ruta explícita al python del entorno virtual
set PYTHON_VENV=%~dp0.venv\Scripts\python.exe

echo Ejecutando el dashboard interactivo...
"%PYTHON_VENV%" "app_dashboard.py"

pause
