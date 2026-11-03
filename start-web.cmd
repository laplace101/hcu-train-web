@echo off
REM Start hcu-train-web on Windows.
REM
REM   start-web.cmd            -> http://127.0.0.1:7766
REM   start-web.cmd 8080 0.0.0.0 -> custom port / bind address

setlocal
set PORT=%1
if "%PORT%"=="" set PORT=7766
set HOST=%2
if "%HOST%"=="" set HOST=127.0.0.1

cd /d "%~dp0"

python -c "import flask" 2>nul
if errorlevel 1 (
    echo [hcu-train-web] installing dependencies...
    python -m pip install -r requirements.txt || goto :fail
)

python -c "import sys; sys.path.insert(0, r'%~dp0..\hcu-train-simulator\src'); import hcu_train_simulator.simulator" 2>nul
if errorlevel 1 (
    echo [hcu-train-web] WARNING: hcu-train-simulator not found next to this project.
    echo                 Expected: %~dp0..\hcu-train-simulator
    echo                 Or set HCU_SIM_SRC to its src directory.
)

echo [hcu-train-web] serving on http://%HOST%:%PORT%
python app.py --host %HOST% --port %PORT%
goto :eof

:fail
echo [hcu-train-web] dependency installation failed.
exit /b 1