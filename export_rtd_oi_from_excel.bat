@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%export_rtd_oi_from_excel.ps1"
set "DEFAULT_WORKBOOK_PATH=%SCRIPT_DIR%RTD_OI_BOVA11.xlsx"

if not exist "%PS_SCRIPT%" (
    echo PowerShell exporter not found:
    echo %PS_SCRIPT%
    exit /b 1
)

set "WORKBOOK_PATH=%~1"
set "WORKSHEET_NAME=%~2"
set "WAIT_SECONDS=%~3"

if "%WORKBOOK_PATH%"=="" if exist "%DEFAULT_WORKBOOK_PATH%" set "WORKBOOK_PATH=%DEFAULT_WORKBOOK_PATH%"
if "%WORKSHEET_NAME%"=="" set "WORKSHEET_NAME=RTD_OI"
if "%WAIT_SECONDS%"=="" set "WAIT_SECONDS=8"

if "%WORKBOOK_PATH%"=="" (
    set /p "WORKBOOK_PATH=Enter full Excel workbook path: "
)

if "%WORKBOOK_PATH%"=="" (
    echo No workbook path provided.
    exit /b 1
)

if not exist "%WORKBOOK_PATH%" (
    echo Workbook not found:
    echo %WORKBOOK_PATH%
    exit /b 1
)

echo Running RTD OI export...
echo Workbook: %WORKBOOK_PATH%
echo Worksheet: %WORKSHEET_NAME%
echo Wait Seconds: %WAIT_SECONDS%

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -WorkbookPath "%WORKBOOK_PATH%" -WorksheetName "%WORKSHEET_NAME%" -WaitSeconds %WAIT_SECONDS%
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo Export failed with exit code %EXIT_CODE%.
    exit /b %EXIT_CODE%
)

echo Export complete.
exit /b 0