@echo off
chcp 65001 >nul
REM One-command site runner (Windows): single-site 5-fold training, then the FL client (iterates over folds x methods automatically).
REM Usage: run_center.bat <exp> <data_root> <site> [run_name|-] [<spacing_file>|-] [<label_map_json>|-]
REM Each run is saved separately under outputs\<exp>\<run>\ (default run name = start time). To resume, pass the same run name as the 4th argument.
REM To stop: create outputs\<exp>\<run>\client_<site>\STOP.txt (exits cleanly after the current session). On restart, completed folds/methods are skipped.
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0\.."
for /f %%f in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive -Name (Get-Location).Drive.Name).Free/1GB)"') do set FREE_GB=%%f
if %FREE_GB% LSS 25 (echo [ERROR] Free disk space %FREE_GB%GB ^< 25GB - free up space and rerun ^(this experiment needs about 60-120GB per site^) & exit /b 1)
if %FREE_GB% LSS 100 echo [WARNING] Free disk space %FREE_GB%GB - may run out during the experiment ^(120GB+ recommended^)
if "%~3"=="" (echo usage: run_center.bat exp4c ^<data_root^> A & exit /b 1)
set "EXP=%~1"
set "DATA=%~2"
set "SITE=%~3"
set "RUN=%~4"
set "SPACING=%~5"
set "LABELMAP=%~6"
if "%SPACING%"=="-" set SPACING=
if "%LABELMAP%"=="-" set LABELMAP=
set SPACING_ARG=
if not "%SPACING%"=="" set SPACING_ARG=--spacing-file "%SPACING%"
set LABEL_ARG=
if not "%LABELMAP%"=="" set LABEL_ARG=--lesion-labels "%LABELMAP%"
if "%RUN%"=="" set RUN=%RUN_NAME%
if "%RUN%"=="" for /f "tokens=1-3 delims=/:. " %%a in ("%date% %time%") do set RUN=run_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set RUN=%RUN: =0%
set PYTHONIOENCODING=utf-8
REM Disable console QuickEdit (clicking pauses the process) and prevent sleep
powershell -NoProfile -Command "$s=(Get-ItemProperty HKCU:\Console).QuickEdit; if($s -ne 0){Set-ItemProperty HKCU:\Console QuickEdit 0}" >nul 2>nul
powercfg /change standby-timeout-ac 0 >nul 2>nul
if not exist outputs\%EXP%\%RUN%\client_%SITE% mkdir outputs\%EXP%\%RUN%\client_%SITE%
echo %RUN%> outputs\%EXP%\LAST_RUN
set LOG=outputs\%EXP%\%RUN%\client_%SITE%\run_center.log
REM Pre-run GPU check (prevents CPU fallback)
.venv\Scripts\python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>>%LOG%
if errorlevel 1 (echo [ERROR] No GPU detected. Check nvidia-smi and rerun scripts\install.bat ^(do not use uv sync/uv run^) & echo GPU FAIL >> %LOG% & exit /b 1)
REM Server connectivity pre-check: the FL port is first contacted only after single-site training ends, so check it up front (set SKIP_SERVER_CHECK=1 to skip)
if not "%SKIP_SERVER_CHECK%"=="1" (
  .venv\Scripts\python.exe scripts\check_server.py --exp %EXP%
  if errorlevel 1 (echo [ERROR] Server connectivity check failed - verify the server is running, the firewall, and the SERVER_IP substitution, then rerun & echo SERVER CHECK FAIL >> %LOG% & exit /b 1)
)
REM Fetch pretrained weights automatically
for /f "tokens=2" %%w in ('findstr /b "init_weights:" configs\%EXP%.yaml') do set IW=%%w
set IW=%IW:"=%
for /f "tokens=2" %%u in ('findstr /b "upload_url:" configs\%EXP%.yaml') do set UURL=%%u
set UURL=%UURL:"=%
set UURL=%UURL:/upload=%
if not "%IW%"=="" if not exist "%IW%" (
  for %%F in ("%IW%") do if not exist "%%~dpF" mkdir "%%~dpF" 2>nul
  echo [WEIGHTS] %IW% not found - downloading from %UURL%/weights >> %LOG%
  curl -fsSL -o "%IW%" "%UURL%/weights" || (del "%IW%" 2>nul & echo [ERROR] Initialization weights not available from the server - create %IW% with scripts\pretrain.py, see README & exit /b 1)
)
echo [%date% %time%] start %EXP% %SITE% >> %LOG%
:loop
if exist outputs\%EXP%\%RUN%\client_%SITE%\STOP.txt (echo STOP.txt - exit >> %LOG% & goto end)
.venv\Scripts\python scripts\patient_catalog.py --data "%DATA%" --exp %EXP% --site %SITE% --run %RUN% %SPACING_ARG% %LABEL_ARG% >> %LOG% 2>&1
set RS=1
for /f "tokens=2" %%r in ('findstr /b "run_single:" configs\%EXP%.yaml') do set RS=%%r
if "%RS%"=="0" goto fl
.venv\Scripts\python scripts\single.py --config configs\%EXP%.yaml --data "%DATA%" --site %SITE% --run %RUN% %SPACING_ARG% >> %LOG% 2>&1
findstr /c:"SINGLE_DONE" outputs\%EXP%\%RUN%\client_%SITE%\single.log >nul || (echo [%date% %time%] single not finished - retry in 60s >> %LOG% & timeout /t 60 /nobreak >nul & goto loop)
:fl
REM Send the true local (single-site) models and metrics to the server before FL starts
.venv\Scripts\python scripts\export_results.py --exp %EXP% --site %SITE% --run %RUN% >> %LOG% 2>&1 && .venv\Scripts\python scripts\upload_results.py --exp %EXP% --site %SITE% --run %RUN% >> %LOG% 2>&1
.venv\Scripts\python scripts\client.py --config configs\%EXP%.yaml --data "%DATA%" --site %SITE% --run %RUN% %SPACING_ARG% >> %LOG% 2>&1
if exist outputs\%EXP%\%RUN%\client_%SITE%\STOP.txt goto end
findstr /c:"ALL_SESSIONS_DONE" %LOG% >nul && goto end
echo [%date% %time%] process exited unexpectedly - restart in 60s >> %LOG%
timeout /t 60 /nobreak >nul
goto loop
:end
REM After completion: anonymized export, then upload to the server (on failure only the zip is kept)
.venv\Scripts\python scripts\export_results.py --exp %EXP% --site %SITE% --run %RUN% >> %LOG% 2>&1
.venv\Scripts\python scripts\upload_results.py --exp %EXP% --site %SITE% --run %RUN% >> %LOG% 2>&1
echo [%date% %time%] done >> %LOG%
endlocal
