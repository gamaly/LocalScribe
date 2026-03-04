#!/bin/bash
# macOS: double-click this file in Finder to launch Whisper Transcribe
cd "$(dirname "$0")"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

python -m whisper_transcribe.app
