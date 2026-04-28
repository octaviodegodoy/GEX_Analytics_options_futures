@echo off
REM Batch file to run daily BOVA11 GEX report
cd /d "%~dp0"
REM Activate your Python environment if needed (uncomment and edit the next line)
REM call path\to\venv\Scripts\activate.bat
python daily_bova11_gex_report.py
