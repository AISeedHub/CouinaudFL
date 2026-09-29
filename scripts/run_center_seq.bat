@echo off
chcp 65001 >nul
REM Run experiments sequentially with optional auxiliary files: run_center_seq.bat <data_root> <site> [<spacing_file>] [<label_map_json>] exp4c
REM   File arguments are detected in any order (.json = label map, any other existing file = spacing). If omitted, the data folder convention / built-in mapping is used.
cd /d "%~dp0\.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "DATA=%~1"
set "SITE=%~2"
shift & shift
set SPACING=-& set LABELMAP=-& set EXPS=
:scan
if "%~1"=="" goto run
if exist "configs\%~1.yaml" (set EXPS=%EXPS% %~1& goto nextarg)
if exist "%~1" (
  if /i "%~x1"==".json" (set "LABELMAP=%~1") else (set "SPACING=%~1")
  goto nextarg
)
echo Unknown argument: %~1 & exit /b 1
:nextarg
shift
goto scan
:run
if "%EXPS%"=="" (echo No experiment name given & exit /b 1)
for %%E in (%EXPS%) do call scripts\run_center.bat %%E "%DATA%" %SITE% "%RUN_NAME%" "%SPACING%" "%LABELMAP%"
