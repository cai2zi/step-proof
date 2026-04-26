@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "STEP_PROOF_ROOT=%%~fI"
cd /d "%STEP_PROOF_ROOT%"

if not defined PYTHON set "PYTHON=python"
if not defined RESULTS_ROOT set "RESULTS_ROOT=%STEP_PROOF_ROOT%\results"
if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=8765"
if not defined SOURCE set "SOURCE=results"
if not defined GRAPH_ONLY set "GRAPH_ONLY=0"

set "CMD=%PYTHON% %STEP_PROOF_ROOT%\scripts\interactive_stage3_viewer.py --results-root \"%RESULTS_ROOT%\" --host \"%HOST%\" --port \"%PORT%\" --source \"%SOURCE%\""
if /I "%GRAPH_ONLY%"=="1" set "CMD=%CMD% --graph-only"

echo Starting interactive viewer at http://%HOST%:%PORT%
call %CMD%

endlocal
