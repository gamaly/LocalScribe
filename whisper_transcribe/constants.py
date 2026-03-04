"""Shared constants for Whisper Transcribe."""

import platform

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mpga", ".mpeg"}

WHISPER_MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
]

DEFAULT_MODEL = "large-v3"


def get_fonts() -> tuple[str, str]:
    """Return (sans_font, mono_font) appropriate for the current platform."""
    system = platform.system()
    if system == "Windows":
        return ("Segoe UI", "Consolas")
    elif system == "Darwin":
        return ("SF Pro", "Menlo")
    else:
        return ("sans-serif", "monospace")
