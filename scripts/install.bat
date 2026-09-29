@echo off
chcp 65001 >nul
REM Windows install: selects the torch wheel matching the CUDA driver version. No lock file.
cd /d "%~dp0\.."
where uv >nul 2>nul || pip install -q uv
if not exist .venv uv venv --python 3.12 .venv
set CUDA=0.0
for /f "usebackq delims=" %%v in (`powershell -NoProfile -Command "(nvidia-smi | Select-String 'CUDA Version: *([0-9]+\.[0-9]+)').Matches[0].Groups[1].Value"`) do set CUDA=%%v
for /f "tokens=1,2 delims=." %%a in ("%CUDA%") do (set MAJ=%%a& set MIN=%%b)
set IDX=cpu
if %MAJ% GEQ 13 set IDX=cu130
if %MAJ%==12 if %MIN% GEQ 8 set IDX=cu128
if %MAJ%==12 if %MIN% LSS 8 if %MIN% GEQ 6 set IDX=cu126
if %MAJ%==12 if %MIN% LSS 6 set IDX=cu121
echo CUDA driver %CUDA% -^> torch index %IDX%
REM Order matters: dependencies first, torch last with a forced index (prevents PyPI from overwriting it with cu130)
uv pip install --python .venv -e .
uv pip install --python .venv --reinstall torch torchvision --index-url https://download.pytorch.org/whl/%IDX%
uv pip install --python .venv "numpy<2"
.venv\Scripts\python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
