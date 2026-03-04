#!/usr/bin/env python3
"""
Build a standalone executable for Whisper Transcribe using PyInstaller.

Usage:
    python build.py

Produces:
    dist/WhisperTranscribe.exe   (Windows)
    dist/WhisperTranscribe.app   (macOS)

Requires: pip install pyinstaller
"""

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_NAME = "WhisperTranscribe"


def find_customtkinter_data():
    """Locate customtkinter's package directory for bundling its assets."""
    try:
        import customtkinter
        pkg_dir = Path(customtkinter.__file__).resolve().parent
        return str(pkg_dir), "customtkinter"
    except ImportError:
        print("ERROR: customtkinter not installed. Run: pip install customtkinter")
        sys.exit(1)


def build():
    ctk_src, ctk_dest = find_customtkinter_data()
    system = platform.system()

    # Create a minimal entry-point script for PyInstaller
    entry_script = ROOT / "_build_entry.py"
    entry_script.write_text(
        "from whisper_transcribe.app import run_app\nrun_app()\n",
        encoding="utf-8",
    )

    try:
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--name", APP_NAME,
            "--windowed",                  # No console window
            "--noconfirm",                 # Overwrite previous build
            "--clean",                     # Clean cache before building
            # Bundle customtkinter theme/assets
            "--add-data", f"{ctk_src}{os.pathsep}{ctk_dest}",
            # Hidden imports that PyInstaller misses
            "--hidden-import", "whisper",
            "--hidden-import", "torch",
            "--hidden-import", "torchaudio",
            "--hidden-import", "tqdm",
            "--hidden-import", "customtkinter",
            # Entry point
            str(entry_script),
        ]

        if system == "Darwin":
            cmd.extend(["--osx-bundle-identifier", "com.whispertranscribe.app"])

        print(f"Building {APP_NAME} for {system}...")
        print(f"Command: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, cwd=str(ROOT))
    finally:
        # Clean up temporary entry script
        entry_script.unlink(missing_ok=True)

    if result.returncode == 0:
        if system == "Windows":
            exe_path = ROOT / "dist" / f"{APP_NAME}.exe"
            folder_path = ROOT / "dist" / APP_NAME / f"{APP_NAME}.exe"
            found = exe_path if exe_path.exists() else folder_path
            print(f"\nBuild complete: {found}")
        elif system == "Darwin":
            print(f"\nBuild complete: {ROOT / 'dist' / f'{APP_NAME}.app'}")
        else:
            print(f"\nBuild complete: {ROOT / 'dist' / APP_NAME}")
        print("\nNote: Whisper models are NOT bundled. They will be downloaded")
        print("automatically on first use (~1-3 GB depending on model size).")
    else:
        print(f"\nBuild failed with exit code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    build()
