#!/usr/bin/env python3
"""
Transcribe audio files locally using OpenAI Whisper.

Supports NVIDIA GPUs (CUDA), Apple Silicon (MPS), and CPU fallback.
"""

import argparse
import sys
from pathlib import Path

import torch
import whisper

from whisper_transcribe.constants import AUDIO_EXTENSIONS, DEFAULT_MODEL, WHISPER_MODELS


def get_device(force_cuda: bool = True):
    """Pick the best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {name}", file=sys.stderr)
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS).", file=sys.stderr)
        return "mps"

    if force_cuda:
        msg = (
            "No GPU available (CUDA not found, MPS not found). "
            "Enable 'Use CPU if GPU not available' in Options, or install PyTorch with GPU support.\n"
            "  NVIDIA: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
            "  Apple Silicon: pip install torch torchaudio  (MPS is built-in)"
        )
        raise RuntimeError(msg)

    print("Using CPU (slower).", file=sys.stderr)
    return "cpu"


def transcribe_file(
    model,
    audio_path: Path,
    output_dir: Path | None,
    output_txt: bool = True,
    output_srt: bool = False,
    language: str | None = None,
    task: str = "transcribe",
    fp16: bool = True,
) -> str:
    """Transcribe one audio file and optionally save to disk."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    result = model.transcribe(
        str(audio_path),
        language=language or None,
        task=task,
        fp16=fp16,
        verbose=False,
    )
    text = result["text"].strip()

    out_dir = output_dir or audio_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / audio_path.stem

    if output_txt:
        txt_path = base.with_suffix(".txt")
        txt_path.write_text(text, encoding="utf-8")
        print(f"  -> {txt_path}")

    if output_srt and result.get("segments"):
        srt_path = base.with_suffix(".srt")
        srt_lines = segments_to_srt(result["segments"])
        srt_path.write_text(srt_lines, encoding="utf-8")
        print(f"  -> {srt_path}")

    return text


def segments_to_srt(segments: list[dict]) -> str:
    """Convert Whisper segments to SRT subtitle format."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = format_srt_time(seg["start"])
        end = format_srt_time(seg["end"])
        text = seg.get("text", "").strip()
        blocks.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def format_srt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def run_transcription(
    files: list[Path],
    output_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    output_srt: bool = False,
    language: str | None = None,
    translate: bool = False,
    fp16: bool = True,
    cpu: bool = False,
    allow_cpu_fallback: bool = False,
    cancelled=None,
    log=None,
):
    """Run transcription on a list of files.

    Optional log(str) callback, cancelled() callable to stop between files.
    """
    if log is None:
        def log(s):
            print(s, file=sys.stderr)

    if not files:
        log("No audio files to transcribe.")
        return

    if cpu:
        device = "cpu"
    else:
        device = get_device(force_cuda=not allow_cpu_fallback)

    log(f"Using device: {device}")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        log("GPU: Apple Silicon (MPS)")

    log(f"Loading model '{model_name}'...")
    model = whisper.load_model(model_name, device=device)
    task = "translate" if translate else "transcribe"

    # fp16 is only reliable on CUDA
    use_fp16 = fp16 and device == "cuda"

    for i, path in enumerate(files, 1):
        if cancelled and cancelled():
            log("Cancelled.")
            return
        log(f"[{i}/{len(files)}] {path.name}")
        try:
            transcribe_file(
                model,
                path,
                output_dir=output_dir,
                output_txt=True,
                output_srt=output_srt,
                language=language,
                task=task,
                fp16=use_fp16,
            )
        except Exception as e:
            log(f"  Error: {e}")

    log("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio locally with OpenAI Whisper."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="Audio file(s) or directory to transcribe (e.g. .wav, .mp3, .m4a, .flac).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for .txt (and .srt). Default: same as input file.",
    )
    parser.add_argument(
        "-m", "--model",
        choices=WHISPER_MODELS,
        default=DEFAULT_MODEL,
        help=f"Whisper model size. Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--no-txt",
        action="store_true",
        help="Do not write .txt transcript.",
    )
    parser.add_argument(
        "--srt",
        action="store_true",
        help="Also write .srt subtitles with timestamps.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g. en, ar). Auto-detect if not set.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate to English instead of transcribing.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 (use if you get GPU errors).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (no GPU).",
    )
    args = parser.parse_args()

    files: list[Path] = []
    for p in args.input:
        p = Path(p)
        if p.is_file():
            if p.suffix.lower() in AUDIO_EXTENSIONS:
                files.append(p)
            else:
                print(f"Skip (not audio): {p}", file=sys.stderr)
        elif p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in AUDIO_EXTENSIONS:
                    files.append(f)
        else:
            print(f"Not found: {p}", file=sys.stderr)

    if not files:
        print("No audio files to transcribe.", file=sys.stderr)
        sys.exit(1)

    try:
        run_transcription(
            files=files,
            output_dir=args.output_dir,
            model_name=args.model,
            output_srt=args.srt,
            language=args.language,
            translate=args.translate,
            fp16=not args.no_fp16,
            cpu=args.cpu,
            allow_cpu_fallback=not args.cpu,
            log=lambda s: print(s, file=sys.stderr),
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
