"""
Music Suite — YouTube & SoundCloud Downloader + Music Sorter.
"""

import os
import sys
import shutil
import threading
import subprocess
import pathlib
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# ── Color palette ────────────────────────────────────────────────────────────
BG      = "#0d0d0d"
SURFACE = "#141414"
CARD    = "#1a1a1a"
ACCENT  = "#c8ff00"   # electric lime
ACCENT2 = "#00d4ff"   # cyan
TEXT    = "#f0f0f0"
MUTED   = "#666666"
SUCCESS = "#39d353"
ERROR   = "#ff4d4d"
WARN    = "#ffaa00"

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac", ".wma", ".opus", ".aiff"}

# ═══════════════════════════════════════════════════════════════════════════════
#  DOWNLOADER LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def check_ffmpeg():
    common_paths = [
        '/usr/local/bin/ffmpeg', '/opt/homebrew/bin/ffmpeg',
        '/opt/local/bin/ffmpeg', '/usr/bin/ffmpeg',
    ]
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    for path in common_paths:
        if os.path.exists(path):
            try:
                subprocess.run([path, '-version'], capture_output=True, check=True)
                os.environ['PATH'] = os.path.dirname(path) + ':' + os.environ.get('PATH', '')
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
    return False

def download_audio_as_wav(url, output_filename="audio", track_title=None, status_callback=None):
    try:
        import yt_dlp
    except ImportError:
        if status_callback:
            status_callback("Error: yt-dlp not installed. Run: pip install yt-dlp")
        return None

    # Clean YouTube-specific list params, but leave SoundCloud/others alone
    if "youtube.com" in url or "youtu.be" in url:
        url = url.split("&list=")[0]
    
    output_dir = str(pathlib.Path.home()) + "/Desktop/Downloaded Songs/"
    os.makedirs(output_dir, exist_ok=True)
    final_wav = f"{output_dir}{output_filename}.wav"

    if track_title is None:
        track_title = output_filename

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}{output_filename}',
        'postprocessors': [
            {'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'},
            {'key': 'FFmpegMetadata', 'add_metadata': False},
        ],
        'postprocessor_args': [
            '-ar', '44100', '-ac', '2',
            '-map_metadata', '-1',
            '-metadata', f'title={track_title}',
            '-fflags', '+bitexact',
        ],
        'user_agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'cookiesfrombrowser': ('chrome',),
        'quiet': True,
        'no_warnings': True,
    }

    # Only add YouTube-specific args if it's a YouTube link
    if "youtube.com" in url or "youtu.be" in url:
        ydl_opts['extractor_args'] = {'youtube': {'player_client': ['android', 'web']}}

    try:
        if status_callback:
            status_callback(f"Fetching: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if status_callback:
            status_callback(f"✓ Saved: {final_wav}")
        return final_wav
    except Exception as e:
        if status_callback:
            status_callback(f"Error: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO ANALYSIS + CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_audio(filepath):
    try:
        import torch
        import torchaudio
        import torchaudio.functional as F
        import torchaudio.transforms as T

        TARGET_SR   = 22050
        MAX_SAMPLES = TARGET_SR * 60

        waveform, sr = torchaudio.load(filepath)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            waveform = F.resample(waveform, sr, TARGET_SR)
            sr = TARGET_SR
        if waveform.shape[1] > MAX_SAMPLES:
            waveform = waveform[:, :MAX_SAMPLES]

        y   = waveform.squeeze(0)
        rms = float(y.pow(2).mean().sqrt())

        signs = torch.sign(y)
        signs[signs == 0] = 1
        zcr = float((signs[:-1] != signs[1:]).float().mean())

        n_fft = 2048
        hop   = 512
        win   = torch.hann_window(n_fft)
        stft  = torch.stft(y, n_fft=n_fft, hop_length=hop,
                           win_length=n_fft, window=win, return_complex=True)
        magnitude = stft.abs()
        power     = magnitude.pow(2)
        freqs     = torch.linspace(0, sr / 2, magnitude.shape[0])

        freq_col  = freqs.unsqueeze(1)
        mag_sum   = magnitude.sum(dim=0).clamp(min=1e-8)
        sc_frames = (freq_col * magnitude).sum(dim=0) / mag_sum
        spectral_centroid = float(sc_frames.mean())

        cumsum      = power.cumsum(dim=0)
        total_e     = cumsum[-1, :].clamp(min=1e-8)
        threshold   = 0.85 * total_e
        rolloff_idx = (cumsum < threshold.unsqueeze(0)).sum(dim=0)
        rolloff_idx = rolloff_idx.clamp(0, len(freqs) - 1)
        spectral_rolloff = float(freqs[rolloff_idx].mean())

        n_bands   = 6
        bin_edges = torch.linspace(0, magnitude.shape[0], n_bands + 1).long()
        contrasts = []
        for b in range(n_bands):
            band = magnitude[bin_edges[b]:bin_edges[b + 1], :]
            if band.numel() == 0:
                continue
            peak   = band.amax(dim=0).mean()
            valley = band.amin(dim=0).mean()
            contrasts.append(float((peak - valley + 1e-8).log()))
        contrast_mean = float(sum(contrasts) / len(contrasts)) if contrasts else 0.0

        mfcc_transform = T.MFCC(
            sample_rate=sr, n_mfcc=13,
            melkwargs={"n_fft": n_fft, "hop_length": hop, "n_mels": 64},
        )
        mfcc      = mfcc_transform(y.unsqueeze(0)).squeeze(0)
        mfcc_mean = mfcc.mean(dim=1).tolist()

        eps    = 1e-8
        bin_hz = (sr / 2) / magnitude.shape[0]
        A4     = 440.0
        chroma = torch.zeros(12)
        for i in range(1, magnitude.shape[0]):
            freq_hz = i * bin_hz
            if freq_hz <= 0:
                continue
            midi = 12 * torch.log2(torch.tensor(freq_hz / A4)) + 69
            pitch_class = int(midi.item()) % 12
            chroma[pitch_class] += power[i, :].mean()
        chroma_norm = chroma / (chroma.sum() + eps)
        chroma_mean = float(chroma_norm.max())

        log_mag = (magnitude + eps).log()
        flux    = torch.clamp(log_mag[:, 1:] - log_mag[:, :-1], min=0).sum(dim=0)
        min_lag = int(60.0 / 240 * sr / hop)
        max_lag = int(60.0 /  40 * sr / hop)
        max_lag = min(max_lag, flux.shape[0] - 1)
        if max_lag > min_lag and flux.shape[0] > max_lag:
            acorr = torch.zeros(max_lag - min_lag)
            for li, lag in enumerate(range(min_lag, max_lag)):
                acorr[li] = (flux[:flux.shape[0] - lag] * flux[lag:]).sum()
            best_lag = int(acorr.argmax()) + min_lag
            tempo    = float(60.0 / (best_lag * hop / sr))
        else:
            tempo = 120.0

        return {
            "tempo": tempo, "rms": rms,
            "spectral_centroid": spectral_centroid,
            "spectral_rolloff":  spectral_rolloff,
            "zcr": zcr, "mfcc": mfcc_mean,
            "chroma": chroma_mean, "contrast": contrast_mean,
        }
    except Exception:
        return None

def classify_track(features):
    if features is None:
        return "Uncategorized"

    tempo    = features["tempo"]
    rms      = features["rms"]
    sc       = features["spectral_centroid"]
    zcr      = features["zcr"]
    chroma   = features["chroma"]
    contrast = features["contrast"]

    energy = "low" if rms < 0.08 else ("high" if rms > 0.18 else "medium")

    if tempo > 160 and energy == "high" and zcr > 0.12:                              return "Drum & Bass"
    if (135<=tempo<=150 or 65<=tempo<=75) and energy=="high" and sc>4000:             return "Dubstep / Bass Music"
    if 125<=tempo<=145 and energy=="high" and contrast>25 and zcr<0.1:               return "Techno"
    if 115<=tempo<=130 and energy in ("medium","high") and chroma>0.4:               return "House"
    if 130<=tempo<=150 and energy=="high" and chroma>0.5 and sc>3000:                return "Trance"
    if tempo<90 and energy=="low" and sc<1500:                                        return "Lo-Fi / Downtempo"
    if 110<=tempo<=124 and energy=="medium" and sc<2500:                              return "Deep House / Afro-House"
    if tempo<70 and energy=="low" and sc<2000:                                        return "Sleep & Meditation"
    if tempo<90 and energy=="low":                                                    return "Ambient & Chill"
    if tempo<90 and energy=="medium" and chroma<0.4:                                  return "Sad & Melancholic"
    if tempo<110 and energy=="medium" and zcr<0.07 and contrast>20:                  return "Acoustic & Folk"
    if tempo<120 and zcr<0.06 and sc<3000 and contrast>25:                           return "Classical & Orchestral"
    if 70<=tempo<=105 and energy=="medium" and chroma>0.45:                           return "R&B & Soul"
    if 60<=tempo<=100 and energy in ("medium","high") and zcr<0.08:                  return "Hip-Hop & Trap"
    if 80<=tempo<=140 and contrast>22 and chroma>0.42 and zcr<0.1:                   return "Jazz & Blues"
    if 100<=tempo<=135 and energy=="medium":                                          return "Pop"
    if 120<=tempo<=145 and energy in ("medium","high") and sc>3500:                  return "Dance & Electronic"
    if 110<=tempo<=145 and energy=="high" and chroma>0.45:                           return "Upbeat & Happy"
    if tempo>110 and energy=="high" and zcr>0.1:                                     return "Rock & Indie"
    if tempo>130 and energy=="high" and sc>4000:                                     return "EDM & Rave"
    if tempo>130 and energy=="high":                                                  return "Pump-Up"

    if energy == "high":
        if tempo > 150:
            return "Drum & Bass / Jungle" if zcr > 0.12 else "Hardstyle / Rave"
        if 120 <= tempo <= 145:
            if sc > 4000:      return "EDM & Dubstep"
            if chroma > 0.48:  return "Trance & Melodic"
            if zcr < 0.07:     return "Techno / Driving"
            return "Dance & Peak-Time"
        if 105 <= tempo < 120:
            return "Rock & Indie" if zcr > 0.1 else "Power Pop"
        if tempo < 105:
            if zcr > 0.1:    return "Industrial / Heavy Alt"
            if chroma > 0.4: return "Funk & Soul (High Energy)"
            return "Heavy Hip-Hop / Bass"
        return "Energetic"
    return "Relaxing" if energy == "low" else "Pop"

def get_metadata_hints(filepath):
    try:
        from mutagen import File as MutagenFile
        audio = MutagenFile(filepath, easy=True)
        if audio is None:
            return {}
        tags = {}
        for key in ("genre", "mood"):
            val = audio.get(key)
            if val:
                tags[key] = str(val[0]).strip()
        return tags
    except Exception:
        return {}

def get_desktop():
    if sys.platform == "win32":
        import winreg
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
            )
            desktop, _ = winreg.QueryValueEx(key, "Desktop")
            return desktop
        except Exception:
            pass
    return str(pathlib.Path.home() / "Desktop")

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class MusicSuiteApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Music Suite")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(900, 640)

        w, h = 1100, 740
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        self._sort_folder       = tk.StringVar()
        self._sort_status       = tk.StringVar(value="Choose a music folder to get started.")
        self._sort_running      = False
        self._sort_dest_folder  = None
        self._max_per_folder    = tk.IntVar(value=20)
        self._min_per_subfolder = tk.IntVar(value=5)

        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("Lime.Horizontal.TProgressbar",
                        troughcolor=CARD, background=ACCENT,
                        darkcolor=ACCENT, lightcolor=ACCENT,
                        bordercolor=CARD, relief="flat")

        self._build_ui()

    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=28, pady=(22, 0))
        tk.Label(hdr, text="♫", font=("Courier", 26, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left", pady=(0, 3))
        tk.Label(hdr, text=" Music Suite",
                 font=("Courier", 20, "bold"), fg=TEXT, bg=BG).pack(side="left")
        tk.Label(hdr, text="Downloader  ·  Sorter",
                 font=("Courier", 9), fg=MUTED, bg=BG).pack(side="right", anchor="s", pady=(0, 3))

        tk.Frame(self, bg="#222", height=1).pack(fill="x", padx=28, pady=(14, 0))

        main_cont = tk.Frame(self, bg=BG)
        main_cont.pack(fill="both", expand=True, padx=28, pady=(14, 10))
        
        body = tk.Frame(main_cont, bg=BG)
        body.pack(fill="x")
        body.columnconfigure(0, weight=1, uniform="col")
        body.columnconfigure(2, weight=1, uniform="col")

        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew")
        self._build_downloader(left)

        tk.Frame(body, bg="#222", width=1).grid(row=0, column=1, sticky="ns", padx=20)

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=2, sticky="nsew")
        self._build_sorter(right)

        log_section = tk.Frame(main_cont, bg=BG)
        log_section.pack(fill="both", expand=True, pady=(20, 0))
        tk.Frame(log_section, bg="#222", height=1).pack(fill="x", pady=(0, 10))
        
        log_hdr = tk.Frame(log_section, bg=BG)
        log_hdr.pack(fill="x", pady=(0, 5))
        tk.Label(log_hdr, text="UNIFIED ACTIVITY LOG", font=("Courier", 8, "bold"),
                 fg=ACCENT2, bg=BG).pack(side="left")
        tk.Button(log_hdr, text="Clear Log", font=("Courier", 8),
                  bg=BG, fg=MUTED, relief="flat", cursor="hand2", bd=0,
                  activebackground=BG, activeforeground=TEXT,
                  command=lambda: self._clear_text(self._unified_log)).pack(side="right")

        log_outer = tk.Frame(log_section, bg=CARD, padx=2, pady=2)
        log_outer.pack(fill="both", expand=True)

        self._unified_log = tk.Text(
            log_outer, bg=CARD, fg=TEXT, font=("Courier", 9), relief="flat",
            insertbackground=ACCENT, selectbackground=ACCENT, selectforeground="#000",
            wrap="word", state="disabled", padx=12, pady=10, height=8
        )
        sb = tk.Scrollbar(log_outer, command=self._unified_log.yview,
                          bg=CARD, troughcolor=CARD, relief="flat", width=6)
        self._unified_log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._unified_log.pack(fill="both", expand=True)
        
        self._unified_log.tag_configure("ok",    foreground=SUCCESS)
        self._unified_log.tag_configure("err",   foreground=ERROR)
        self._unified_log.tag_configure("warn",  foreground=WARN)
        self._unified_log.tag_configure("info",  foreground=ACCENT2)
        self._unified_log.tag_configure("head",  foreground=ACCENT, font=("Courier", 9, "bold"))
        self._unified_log.tag_configure("muted", foreground=MUTED)

    def _build_downloader(self, parent):
        parent.columnconfigure(0, weight=1)
        tk.Label(parent, text="↓  DOWNLOAD", font=("Courier", 10, "bold"),
                 fg=ACCENT, bg=BG).grid(row=0, column=0, sticky="w", pady=(0, 10))

        # URL card - Updated label
        c1 = tk.Frame(parent, bg=CARD, padx=16, pady=14)
        c1.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        c1.columnconfigure(0, weight=1)
        tk.Label(c1, text="SOURCE URL (YT / SOUNDCLOUD)", font=("Courier", 8, "bold"),
                 fg=ACCENT2, bg=CARD).grid(row=0, column=0, sticky="w", pady=(0, 5))
        self._dl_url = tk.Entry(
            c1, font=("Courier", 11), bg=SURFACE, fg=TEXT,
            insertbackground=ACCENT, relief="flat",
            highlightthickness=1, highlightcolor=ACCENT, highlightbackground=MUTED
        )
        self._dl_url.grid(row=1, column=0, sticky="ew", ipady=7, ipadx=6)

        c2 = tk.Frame(parent, bg=CARD, padx=16, pady=14)
        c2.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        c2.columnconfigure(0, weight=1)
        tk.Label(c2, text="TRACK TITLE", font=("Courier", 8, "bold"),
                 fg=ACCENT2, bg=CARD).grid(row=0, column=0, sticky="w", pady=(0, 5))
        self._dl_title = tk.Entry(
            c2, font=("Courier", 11), bg=SURFACE, fg=TEXT,
            insertbackground=ACCENT, relief="flat",
            highlightthickness=1, highlightcolor=ACCENT, highlightbackground=MUTED
        )
        self._dl_title.grid(row=1, column=0, sticky="ew", ipady=7, ipadx=6)

        self._dl_status = tk.StringVar(value="Files saved to ~/Desktop/Downloaded Songs/")
        tk.Label(parent, textvariable=self._dl_status,
                 font=("Courier", 8), fg=MUTED, bg=BG,
                 wraplength=420, anchor="w", justify="left").grid(
            row=3, column=0, sticky="ew", pady=(2, 0))

        self._dl_btn = self._mkbtn(parent, "↓  Download WAV", self._do_download,
                                   ACCENT, BG, big=True)
        self._dl_btn.grid(row=4, column=0, sticky="ew", pady=(10, 0))

    def _do_download(self):
        url   = self._dl_url.get().strip()
        title = self._dl_title.get().strip()
        if not url:
            messagebox.showwarning("Input Required", "Please enter a URL", parent=self)
            return
        if not title:
            messagebox.showwarning("Input Required", "Please enter a track title", parent=self)
            return

        self._dl_btn.config(state="disabled")
        self._dl_status.set("Downloading… please wait.")

        def worker():
            def cb(msg):
                tag = "ok" if msg.startswith("✓") else "err" if msg.startswith("Error") else "info"
                self._log_msg(f"[DL] {msg}", tag)
                self.after(0, lambda: self._dl_status.set(msg[:80]))

            result = download_audio_as_wav(url, title, title, cb)

            def finish():
                self._dl_btn.config(state="normal")
                if result:
                    self._dl_status.set("✓ Download complete!")
                    self._dl_url.delete(0, tk.END)
                    self._dl_title.delete(0, tk.END)
                else:
                    self._dl_status.set("Download failed — check the log.")
            self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    #  SORTER & HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_sorter(self, parent):
        parent.columnconfigure(0, weight=1)
        tk.Label(parent, text="♻  SORT", font=("Courier", 10, "bold"),
                 fg=ACCENT, bg=BG).grid(row=0, column=0, sticky="w", pady=(0, 10))

        c1 = tk.Frame(parent, bg=CARD, padx=16, pady=14)
        c1.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        c1.columnconfigure(0, weight=1)
        tk.Label(c1, text="SOURCE FOLDER", font=("Courier", 8, "bold"),
                 fg=ACCENT2, bg=CARD).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        self._sort_entry = tk.Entry(
            c1, textvariable=self._sort_folder,
            font=("Courier", 11), bg=SURFACE, fg=TEXT,
            insertbackground=ACCENT, relief="flat",
            highlightthickness=1, highlightcolor=ACCENT, highlightbackground=MUTED
        )
        self._sort_entry.grid(row=1, column=0, sticky="ew", ipady=6, ipadx=6)
        self._mkbtn(c1, "Browse →", self._sort_browse, ACCENT, BG).grid(
            row=1, column=1, padx=(8, 0))

        c2 = tk.Frame(parent, bg=CARD, padx=16, pady=14)
        c2.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        tk.Label(c2, text="FOLDER SPLIT SETTINGS", font=("Courier", 8, "bold"),
                 fg=ACCENT2, bg=CARD).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        tk.Label(c2, text="Split when songs exceed:",
                 font=("Courier", 9), fg=TEXT, bg=CARD).grid(row=1, column=0, sticky="w")
        tk.Spinbox(c2, from_=2, to=9999, textvariable=self._max_per_folder,
                   width=6, font=("Courier", 10), bg=SURFACE, fg=ACCENT,
                   buttonbackground=CARD, relief="flat",
                   highlightthickness=1, highlightcolor=ACCENT, highlightbackground=MUTED,
                   insertbackground=ACCENT).grid(row=1, column=1, padx=(8, 20), sticky="w")
        tk.Label(c2, text="Min per subfolder:",
                 font=("Courier", 9), fg=TEXT, bg=CARD).grid(row=1, column=2, sticky="w")
        tk.Spinbox(c2, from_=1, to=9999, textvariable=self._min_per_subfolder,
                   width=6, font=("Courier", 10), bg=SURFACE, fg=ACCENT,
                   buttonbackground=CARD, relief="flat",
                   highlightthickness=1, highlightcolor=ACCENT, highlightbackground=MUTED,
                   insertbackground=ACCENT).grid(row=1, column=3, padx=(8, 0), sticky="w")

        self._sort_output_label = tk.StringVar(value="Sorted folders saved to your Desktop.")
        tk.Label(parent, textvariable=self._sort_output_label,
                 font=("Courier", 8), fg=MUTED, bg=BG,
                 wraplength=420, anchor="w", justify="left").grid(
            row=3, column=0, sticky="ew", pady=(2, 0))

        self._sort_progress = ttk.Progressbar(
            parent, style="Lime.Horizontal.TProgressbar",
            mode="determinate", length=100)
        self._sort_progress.grid(row=4, column=0, sticky="ew", pady=(6, 0))

        self._sort_start_btn = self._mkbtn(
            parent, "▶  Start Sorting", self._sort_start, ACCENT, BG, big=True)
        self._sort_start_btn.grid(row=5, column=0, sticky="ew", pady=(10, 0))

    def _sort_browse(self):
        folder = filedialog.askdirectory(title="Select Music Folder")
        if folder:
            self._sort_folder.set(folder)
            self._log_msg(f"[Sort] Selected source: {folder}", "info")

    def _sort_start(self):
        if self._sort_running: return
        folder = self._sort_folder.get().strip()
        if not folder or not os.path.isdir(folder):
            self._log_msg("[Sort] Error: Please select a valid music folder first.", "err")
            return
        self._sort_running = True
        self._sort_start_btn.config(state="disabled", text="Processing…")
        self._sort_progress.configure(value=0)
        threading.Thread(target=self._sort_process, args=(folder,), daemon=True).start()

    def _sort_process(self, src_folder):
        try:
            self._log_msg("[Sort] Scanning folder for audio files…", "info")
            files = [f for f in os.listdir(src_folder) if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS]
            if not files:
                self._log_msg("[Sort] No audio files found in the selected folder.", "err")
                return

            self._log_msg(f"[Sort] Found {len(files)} audio file(s). Analysing…", "head")
            folder_name = os.path.basename(src_folder.rstrip("/\\")) + " (Sorted)"
            dest_root   = os.path.join(get_desktop(), folder_name)
            os.makedirs(dest_root, exist_ok=True)
            self.after(0, lambda: self._sort_output_label.set(f"Saved to: Desktop/{folder_name}"))

            max_per_folder    = max(2, self._max_per_folder.get())
            min_per_subfolder = max(1, self._min_per_subfolder.get())
            total             = len(files)
            categorized       = []

            for i, fname in enumerate(files, 1):
                self.after(0, lambda idx=i, t=total: self._sort_progress.configure(value=int(idx / t * 50)))
                fpath    = os.path.join(src_folder, fname)
                features = analyze_audio(fpath)
                category = classify_track(features)
                if features is None:
                    hints = get_metadata_hints(fpath)
                    category = hints.get("genre", "Uncategorized")
                self._log_msg(f"  ✓  {fname}  →  {category}", "ok")
                categorized.append((fpath, fname, category))

            from collections import defaultdict
            groups = defaultdict(list)
            for fpath, fname, category in categorized: groups[category].append((fpath, fname))

            self._log_msg("\n── Copying files ────────────────────", "head")
            total_copy = len(categorized)
            copy_idx   = 0

            for category, tracks in sorted(groups.items()):
                count = len(tracks)
                cat_dir = os.path.join(dest_root, category)
                os.makedirs(cat_dir, exist_ok=True)
                for fpath, fname in tracks:
                    copy_idx += 1
                    dest_file = os.path.join(cat_dir, fname)
                    shutil.copy2(fpath, dest_file)
                    pct = 50 + int(copy_idx / total_copy * 50)
                    self.after(0, lambda p=pct: self._sort_progress.configure(value=p))

            self.after(0, lambda: self._sort_progress.configure(value=100))
            self._log_msg(f"\n✔ Done! {total} tracks sorted.", "head")
        except Exception as e:
            self._log_msg(f"[Sort] Error: {e}", "err")
        finally:
            self._sort_running = False
            self.after(0, lambda: self._sort_start_btn.config(state="normal", text="▶  Start Sorting"))

    def _mkbtn(self, parent, text, cmd, bg, fg, big=False):
        font_size = 11 if big else 9
        pad       = (16, 12) if big else (12, 7)
        return tk.Button(
            parent, text=text, command=cmd, font=("Courier", font_size, "bold"),
            bg=bg, fg=fg, relief="flat", cursor="hand2", activebackground=fg, activeforeground=bg,
            padx=pad[0], pady=pad[1], bd=0
        )

    def _log_msg(self, msg, tag=""):
        def _write():
            self._unified_log.config(state="normal")
            self._unified_log.insert("end", msg + "\n", tag)
            self._unified_log.see("end")
            self._unified_log.config(state="disabled")
        self.after(0, _write)

    def _clear_text(self, widget):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.config(state="disabled")

if __name__ == "__main__":
    if not check_ffmpeg():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("FFmpeg Missing", "FFmpeg is required for downloads.")
        sys.exit(1)
    app = MusicSuiteApp()
    app.mainloop()