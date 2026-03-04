@echo off
title Whisper Transcribe
cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

pythonw -m whisper_transcribe.app
if errorlevel 1 python -m whisper_transcribe.app
if errorlevel 1 pause
