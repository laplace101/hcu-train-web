@echo off
REM Deploy hcu-train-web on a Windows workstation.
REM
REM   deploy.cmd                start / redeploy on 0.0.0.0:7766
REM   deploy.cmd stop
REM   deploy.cmd status
REM   deploy.cmd start 8080
REM
REM Self-contained: hcu-train-simulator sits next to hcu-train-web.

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=start"

set "PORT=7766"
if /i "%ACTION%"=="start" if not "%~2"=="" set "PORT=%~2"

set "PIDFILE=runtime\server.pid"
set "LOGFILE=runtime\server.log"

if /i "%ACTION%"=="stop" goto :stop
if /i "%ACTION%"=="status" goto :status
if /i "%ACTION%"=="restart" goto :stop
if /i not "%ACTION%"=="start" (
    echo [hcu-train-web] unknown action: %ACTION%  ^(use start ^| stop ^| status ^| restart^)
    exit /b 1
)

echo [hcu-train-web] Python:
python --version || goto :nopython

if not exist "app.py" (
    echo [hcu-train-web] app.py not found - run this script from the bundle root.
    exit /b 1
)
if not exist "hcu-train-simulator\src\hcu_train_simulator\__init__.py" (
    echo [hcu-train-web] missing hcu-train-simulator\src\hcu_train_simulator
    exit /b 1
)

call :killold

echo [hcu-train-web] installing dependencies ...
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo [hcu-train-web] dependency installation failed.
    exit /b 1
)

if not exist "runtime" mkdir "runtime"
echo [hcu-train-web] starting on 0.0.0.0:%PORT% ...
start "" /b python -u app.py --host 0.0.0.0 --port %PORT% > "%LOGFILE%" 2>&1

REM health check
set OK=0
for /L %%i in (1,1,30) do (
    timeout /t 1 /nobreak >nul
    curl -fsS "http://127.0.0.1:%PORT%/api/health" >nul 2>&1 && (set OK=1 & goto :healthy)
)
:healthy
if "%OK%"=="1" goto :done
echo [hcu-train-web] service did not become ready. Last log lines:
powershell -NoProfile -Command "Get-Content -Tail 30 '%LOGFILE%'"
exit /b 1

:done
echo.
echo [hcu-train-web] deployment OK
echo   local check   : http://127.0.0.1:%PORT%/
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    for /f "tokens=* delims= " %%b in ("%%a") do echo   LAN access    : http://%%b:%PORT%/
)
echo   log file      : %LOGFILE%
echo.
echo   Open one of the LAN addresses above from your own PC.
echo   If it does not open, allow the port:
echo     netsh advfirewall firewall add rule name="hcu-train-web %PORT%" dir=in action=allow protocol=TCP localport=%PORT%
echo   Or use an SSH tunnel:  ssh -L %PORT%:127.0.0.1:%PORT% user@workstation-ip
exit /b 0

:stop
call :killold
echo [hcu-train-web] stopped.
exit /b 0

:status
set FOUND=0
for /f "tokens=2" %%p in ('tasklist /fi "imagename eq python.exe" /fo csv /nh ^| findstr /c:"python.exe"') do set FOUND=1
if "%FOUND%"=="1" (echo running & exit /b 0) else (echo not running & exit /b 1)

:nopython
echo [hcu-train-web] python not found in PATH.
exit /b 1

:killold
for /f "tokens=2" %%p in ('wmic process where "name='python.exe'" get processid^,commandline /format:csv ^| findstr /i "app.py"') do (
    taskkill /pid %%p /f >nul 2>&1
)
exit /b 0