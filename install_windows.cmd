@echo off
setlocal
rem Complete Windows environment: Python, runtime, algorithms, development tools and GUI.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_env.ps1" -WithGui %*
set "MOBO_SETUP_EXIT=%ERRORLEVEL%"
if not "%MOBO_SETUP_EXIT%"=="0" (
  echo.
  echo Installation failed. Check the error above before retrying.
)
if not defined MOBO_SETUP_NO_PAUSE pause
exit /b %MOBO_SETUP_EXIT%
