# Whisper Transcribe

Transcribe audio files locally using [OpenAI Whisper](https://github.com/openai/whisper). No cloud services, no API keys — everything runs on your machine.

Supports **NVIDIA GPUs** (CUDA), **Apple Silicon** (MPS), and **CPU** fallback.

## Features

- Desktop GUI app — just double-click to launch
- Standalone executables for Windows and macOS (no Python required)
- Batch transcription (files or whole folders)
- SRT subtitle generation with timestamps
- Multiple Whisper model sizes (tiny → large-v3-turbo)
- Auto-detects the best available device (GPU → CPU)

## Download

Grab the latest release for your platform from the [Releases](https://github.com/user/whisper-transcribe/releases) page:

- **Windows**: `WhisperTranscribe.exe`
- **macOS**: `WhisperTranscribe.app`

Double-click to launch. No installation or Python required.

> Whisper models (~1–3 GB) are downloaded automatically on first use and cached locally.

## Usage

In the app you can:
- Add audio files or a whole folder
- Choose the Whisper model (large-v3 for quality, base for speed)
- Set an optional output folder (or save next to each file)
- Optionally create .srt subtitles
- Watch progress in the log

## Building from source

If you prefer to run from source or build the executable yourself:

### 1. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install PyTorch

PyTorch installation depends on your hardware. Pick one:

**NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Apple Silicon / CPU-only:**
```bash
pip install torch torchaudio
```

For other configurations, use the [PyTorch install selector](https://pytorch.org/get-started/locally/).

### 3. Install Whisper Transcribe

```bash
pip install -e .
```

### 4. Run the app

**Windows:** Double-click `Whisper Transcribe.bat`

**macOS:** Double-click `Whisper Transcribe.command`

Or from a terminal:

```bash
whisper-transcribe
```

### 5. Build a standalone executable

```bash
pip install -e ".[dev]"
python build.py
```

The executable will be in the `dist/` folder.

<details>
<summary>CLI (optional)</summary>

A CLI is also available for scripting:

```bash
whisper-transcribe-cli recording.mp3
whisper-transcribe-cli ~/Recordings -o ~/Transcripts --srt -m base
```

| Flag | Description |
|------|-------------|
| `-o, --output-dir DIR` | Save transcripts in DIR (default: same folder as audio) |
| `-m, --model NAME` | Model size: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`, `large-v3-turbo` (default: `large-v3`) |
| `--srt` | Also write .srt subtitle files with timestamps |
| `--language CODE` | Force language (e.g. `en`, `ar`). Omit for auto-detect |
| `--translate` | Translate to English instead of transcribing |
| `--no-fp16` | Disable fp16 on GPU (use if you get GPU errors) |
| `--cpu` | Force CPU mode (no GPU) |

</details>

## Models

| Model | Parameters | Speed | Quality |
|-------|-----------|-------|---------|
| `tiny` | 39M | Fastest | Basic |
| `base` | 74M | Fast | Good |
| `small` | 244M | Moderate | Better |
| `medium` | 769M | Slow | Great |
| `large-v3` | 1550M | Slowest | Best |
| `large-v3-turbo` | 809M | Fast | Near-best |

Models are downloaded automatically on first use and cached locally.

## Supported audio formats

`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`, `.mpga`, `.mpeg`

## License

[MIT](LICENSE)
