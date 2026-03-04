#!/usr/bin/env python3
"""
Desktop app for transcribing audio with Whisper.
Supports NVIDIA GPUs (CUDA), Apple Silicon (MPS), and CPU fallback.
"""

import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from whisper_transcribe.constants import AUDIO_EXTENSIONS, DEFAULT_MODEL, WHISPER_MODELS, get_fonts

# Detect if running as a PyInstaller bundle
FROZEN = getattr(sys, "frozen", False)

SCRIPT_DIR = Path(__file__).resolve().parent.parent
REQUIREMENTS_FILE = SCRIPT_DIR / "requirements.txt"

FONT_SANS, FONT_MONO = get_fonts()


def is_audio(p: Path) -> bool:
    return p.suffix.lower() in AUDIO_EXTENSIONS


def _run_install_then_restart(root):
    """Show 'Install now?' dialog; if OK, run pip install and offer to restart.

    Only used when running from source (not from a bundled executable).
    """
    if not messagebox.askyesno(
        "Packages required",
        "Whisper and other required packages are not installed.\n\n"
        "Click OK to install them now (this may take a few minutes).\n"
        "Click Cancel to exit.",
    ):
        root.destroy()
        sys.exit(0)
        return

    win = tk.Tk()
    win.title("Installing")
    win.geometry("520x320")
    tk.Label(win, text="Installing packages...", font=(FONT_SANS, 11)).pack(pady=(12, 4))
    log = scrolledtext.ScrolledText(win, height=14, wrap=tk.WORD, font=(FONT_MONO, 9))
    log.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
    log_queue: queue.Queue = queue.Queue()
    win.lift()
    win.focus_force()

    def run_pip():
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
                cwd=str(SCRIPT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                log_queue.put(line.rstrip())
            proc.wait()
            log_queue.put(("__done__", proc.returncode))
        except Exception as e:
            log_queue.put(str(e))
            log_queue.put(("__done__", -1))

    def poll_log():
        try:
            while True:
                msg = log_queue.get_nowait()
                if isinstance(msg, tuple) and msg[0] == "__done__":
                    code = msg[1]
                    win.after(0, lambda c=code: finish_install(c))
                    return
                log.insert(tk.END, msg + "\n")
                log.see(tk.END)
        except queue.Empty:
            pass
        win.after(200, poll_log)

    def finish_install(returncode):
        win.destroy()
        root.destroy()
        if returncode == 0:
            if messagebox.askyesno("Done", "Installation complete. Restart the app now?"):
                subprocess.Popen(
                    [sys.executable, "-m", "whisper_transcribe.app"],
                    cwd=str(SCRIPT_DIR),
                )
        else:
            messagebox.showerror(
                "Install failed",
                "Installation had errors. Try running in a terminal:\npip install -r requirements.txt",
            )
        sys.exit(0)

    threading.Thread(target=run_pip, daemon=True).start()
    win.after(200, poll_log)
    win.mainloop()


def run_app():
    root = tk.Tk()
    root.withdraw()

    try:
        import customtkinter as ctk
    except ImportError:
        if FROZEN:
            messagebox.showerror("Error", "Required packages are missing from the build.")
            sys.exit(1)
        _run_install_then_restart(root)
        return

    try:
        from whisper_transcribe.transcribe import run_transcription
    except ImportError:
        if FROZEN:
            messagebox.showerror("Error", "Required packages are missing from the build.")
            sys.exit(1)
        _run_install_then_restart(root)
        return

    root.destroy()

    class QueueWriter:
        """File-like object that pushes lines to a queue (for redirecting stdout/stderr to GUI)."""

        def __init__(self, q: queue.Queue):
            self._q = q
            self._buf = ""

        def write(self, s: str):
            if not isinstance(s, str):
                s = str(s)
            self._buf += s
            while "\n" in self._buf or "\r" in self._buf:
                idx = len(self._buf)
                for sep in ("\n", "\r"):
                    pos = self._buf.find(sep)
                    if pos != -1 and pos < idx:
                        idx = pos
                line = self._buf[:idx].strip()
                # Skip the separator character
                self._buf = self._buf[idx + 1:]
                if line:
                    self._q.put(line)

        def flush(self):
            if self._buf.strip():
                self._q.put(self._buf.strip())
                self._buf = ""

    class App:
        def __init__(self):
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")

            self.root = ctk.CTk()
            self.root.title("Whisper Transcribe")
            self.root.minsize(560, 520)
            self.root.geometry("640x560")

            self.file_list: list[Path] = []
            self.log_queue: queue.Queue = queue.Queue()
            self.running = False
            self.cancelled = False

            self.main = ctk.CTkFrame(self.root, fg_color="transparent")
            self.main.pack(fill="both", expand=True, padx=20, pady=20)

            self.build_ui()
            self.root.after(100, self.poll_log)

        def build_ui(self):
            # Title
            title = ctk.CTkLabel(
                self.main, text="Audio transcription",
                font=ctk.CTkFont(size=22, weight="bold"),
                text_color=("gray10", "gray90"),
            )
            title.pack(anchor="w", pady=(0, 16))

            # Files section
            files_label = ctk.CTkLabel(
                self.main, text="Audio files",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=("gray30", "gray70"),
            )
            files_label.pack(anchor="w", pady=(0, 6))

            list_frame = ctk.CTkFrame(self.main, fg_color=("gray90", "gray17"), corner_radius=8, height=100)
            list_frame.pack(fill="x", pady=(0, 8))
            list_frame.pack_propagate(False)

            lb_bg = "#1a1a2e" if ctk.get_appearance_mode() == "Dark" else "#e8e8e8"
            lb_fg = "#eaeaea" if ctk.get_appearance_mode() == "Dark" else "#1a1a1a"
            self.listbox = tk.Listbox(
                list_frame,
                height=4,
                selectmode=tk.EXTENDED,
                bg=lb_bg,
                fg=lb_fg,
                selectbackground="#0d7377",
                selectforeground="white",
                font=(FONT_SANS, 10),
                relief="flat",
                highlightthickness=0,
                border=0,
            )
            scroll = tk.Scrollbar(list_frame, command=self.listbox.yview, bg=lb_bg)
            self.listbox.config(yscrollcommand=scroll.set)
            self.listbox.pack(side="left", fill="both", expand=True, padx=8, pady=8)
            scroll.pack(side="right", fill="y", pady=8, padx=(0, 8))

            btn_f = ctk.CTkFrame(self.main, fg_color="transparent")
            btn_f.pack(fill="x", pady=(0, 16))
            ctk.CTkButton(btn_f, text="Add file(s)...", width=110, command=self.add_files).pack(side="left", padx=(0, 8))
            ctk.CTkButton(btn_f, text="Add folder...", width=110, command=self.add_folder).pack(side="left", padx=(0, 8))
            ctk.CTkButton(btn_f, text="Remove selected", width=110, command=self.remove_selected, fg_color=("gray65", "gray35")).pack(side="left", padx=(0, 8))
            ctk.CTkButton(btn_f, text="Clear all", width=80, command=self.clear_list, fg_color=("gray65", "gray35")).pack(side="left")

            # Options card
            opt = ctk.CTkFrame(self.main, fg_color=("gray92", "gray20"), corner_radius=10)
            opt.pack(fill="x", pady=(0, 16))
            opt_inner = ctk.CTkFrame(opt, fg_color="transparent")
            opt_inner.pack(fill="x", padx=16, pady=16)

            opt_title = ctk.CTkLabel(opt_inner, text="Options", font=ctk.CTkFont(size=13, weight="bold"), text_color=("gray20", "gray80"))
            opt_title.pack(anchor="w", pady=(0, 12))

            row1 = ctk.CTkFrame(opt_inner, fg_color="transparent")
            row1.pack(fill="x", pady=4)
            ctk.CTkLabel(row1, text="Model", width=90).pack(side="left", padx=(0, 8))
            self.model_var = ctk.StringVar(value=DEFAULT_MODEL)
            self.model_combo = ctk.CTkComboBox(row1, values=WHISPER_MODELS, variable=self.model_var, width=160, state="readonly")
            self.model_combo.pack(side="left")

            self.srt_var = ctk.StringVar(value="off")
            ctk.CTkCheckBox(opt_inner, text="Also create .srt subtitles", variable=self.srt_var, onvalue="on", offvalue="off").pack(anchor="w", pady=8)

            row2 = ctk.CTkFrame(opt_inner, fg_color="transparent")
            row2.pack(fill="x", pady=4)
            ctk.CTkLabel(row2, text="Output folder", width=90).pack(side="left", padx=(0, 8))
            self.out_dir_var = ctk.StringVar(value="")
            ctk.CTkEntry(row2, textvariable=self.out_dir_var, placeholder_text="Same as each file", height=32).pack(side="left", fill="x", expand=True, padx=(0, 8))
            ctk.CTkButton(row2, text="Browse...", width=80, command=self.browse_output).pack(side="left")

            self.allow_cpu_var = ctk.StringVar(value="on")
            ctk.CTkCheckBox(opt_inner, text="Use CPU if GPU not available", variable=self.allow_cpu_var, onvalue="on", offvalue="off").pack(anchor="w", pady=(8, 0))

            # Run / Cancel
            run_f = ctk.CTkFrame(self.main, fg_color="transparent")
            run_f.pack(fill="x", pady=16)
            self.run_btn = ctk.CTkButton(run_f, text="Transcribe", command=self.start_transcribe, height=40, font=ctk.CTkFont(size=14, weight="bold"), width=140)
            self.run_btn.pack(side="left", padx=(0, 12))
            self.cancel_btn = ctk.CTkButton(run_f, text="Cancel", command=self.cancel_transcribe, state="disabled", height=40, fg_color=("gray55", "gray45"), width=100)
            self.cancel_btn.pack(side="left")

            # Log
            log_label = ctk.CTkLabel(self.main, text="Log", font=ctk.CTkFont(size=13, weight="bold"), text_color=("gray30", "gray70"))
            log_label.pack(anchor="w", pady=(0, 6))

            self.log_text = ctk.CTkTextbox(
                self.main,
                height=140,
                font=ctk.CTkFont(family=FONT_MONO, size=11),
                fg_color=("gray95", "gray13"),
                corner_radius=8,
                state="disabled",
            )
            self.log_text.pack(fill="both", expand=True)

        def log(self, msg: str):
            self.log_queue.put(msg)

        def poll_log(self):
            try:
                while True:
                    msg = self.log_queue.get_nowait()
                    self.log_text.configure(state="normal")
                    self.log_text.insert("end", msg + "\n")
                    self.log_text.see("end")
                    self.log_text.configure(state="disabled")
            except queue.Empty:
                pass
            self.root.after(100, self.poll_log)

        def add_files(self):
            paths = filedialog.askopenfilenames(
                title="Select audio files",
                filetypes=[
                    ("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.webm *.mpga *.mpeg"),
                    ("All files", "*.*"),
                ],
            )
            for p in paths:
                p = Path(p)
                if is_audio(p) and p not in self.file_list:
                    self.file_list.append(p)
                    self.listbox.insert(tk.END, p.name)

        def add_folder(self):
            path = filedialog.askdirectory(title="Select folder with audio files")
            if not path:
                return
            added = 0
            for f in sorted(Path(path).iterdir()):
                if f.is_file() and is_audio(f) and f not in self.file_list:
                    self.file_list.append(f)
                    self.listbox.insert(tk.END, f.name)
                    added += 1
            if added:
                self.log(f"Added {added} file(s) from folder.")

        def remove_selected(self):
            for i in reversed(self.listbox.curselection()):
                self.listbox.delete(i)
                self.file_list.pop(i)

        def clear_list(self):
            self.listbox.delete(0, tk.END)
            self.file_list.clear()

        def browse_output(self):
            path = filedialog.askdirectory(title="Output folder (leave empty = same as each file)")
            if path:
                self.out_dir_var.set(path)

        def cancel_transcribe(self):
            if self.running:
                self.cancelled = True
                self.log("Cancelling after current file...")

        def start_transcribe(self):
            if self.running:
                return
            files = list(self.file_list)
            if not files:
                messagebox.showinfo("No files", "Add at least one audio file.")
                return

            out_dir = self.out_dir_var.get().strip()
            output_dir = Path(out_dir) if out_dir else None

            def work():
                self.running = True
                self.cancelled = False
                self.run_btn.configure(state="disabled")
                self.cancel_btn.configure(state="normal")
                writer = QueueWriter(self.log_queue)
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = writer
                error_occurred = False
                try:
                    run_transcription(
                        files=files,
                        output_dir=output_dir,
                        model_name=self.model_var.get(),
                        output_srt=self.srt_var.get() == "on",
                        language=None,
                        translate=False,
                        fp16=True,
                        cpu=False,
                        allow_cpu_fallback=self.allow_cpu_var.get() == "on",
                        cancelled=lambda: self.cancelled,
                        log=self.log,
                    )
                except RuntimeError as e:
                    error_occurred = True
                    self.log(f"Error: {e}")
                    self.root.after(0, lambda err=str(e): messagebox.showerror(
                        "No GPU",
                        err + "\n\nEnable 'Use CPU if GPU not available' in Options to run on CPU.",
                    ))
                except Exception as e:
                    error_occurred = True
                    self.log(f"Error: {e}")
                    self.root.after(0, lambda err=str(e): messagebox.showerror("Error", err))
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                    writer.flush()
                    self.running = False
                    self.run_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    if self.cancelled:
                        self.root.after(0, lambda: messagebox.showinfo("Cancelled", "Transcription was cancelled."))
                    elif not error_occurred:
                        self.root.after(0, lambda: messagebox.showinfo("Done", "Transcription finished. Check the log above."))

            threading.Thread(target=work, daemon=True).start()

        def run(self):
            self.root.mainloop()

    app = App()
    app.run()


if __name__ == "__main__":
    run_app()
