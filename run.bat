@echo off
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv not found in this folder.
  echo Create it first:
  echo   py -3.13 -m venv .venv
  echo   .venv\Scripts\python -m pip install -r requirements.txt
  exit /b 1
)
".venv\Scripts\python.exe" -m ocrtrans.app %*
