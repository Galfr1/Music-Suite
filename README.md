# 🎵 Music Suite

**Music Suite** is a desktop application that lets you:

- 🎧 Download audio from YouTube & SoundCloud as WAV files  
- 🧠 Automatically analyze and classify music  
- ♻️ Organize your music library into categorized folders  

Built with a modern Tkinter UI and powered by Python audio processing tools.

---

## 🐍 Requirements

- **Python 3.11** (recommended)
- **FFmpeg** (required for audio processing)

### Install FFmpeg

- macOS (Homebrew):
  ```bash
  brew install ffmpeg
- Ubuntu:
  ```bash
  sudo apt install ffmpeg
- Windows:
Download from https://ffmpeg.org and add to PATH

### 📦 Python Dependencies

Install required packages with:

  ```bash
  pip install yt-dlp torch torchaudio mutagen pyinstaller
  ```
## 🚀 Running the App (Development)
```bash
python3 Music_Suite.py
```
## 🏗️ Building the App

This project uses PyInstaller to create a standalone app.
Build Command:

  ```bash
  ./build.sh
  ```
### What it does?

Cleans previous builds (build/, dist/)
Uses the provided Music_Suite.spec
Outputs the app to:
dist/Music Suite.app

## 📁 Project Structure
Music_Suite.py     # Main application
Music_Suite.spec   # PyInstaller config
build.sh           # Build script
README.md          # This file

## ⚠️ Notes
FFmpeg must be installed and accessible in PATH
Downloads are saved to:
~/Desktop/Downloaded Songs/
Sorted music folders are saved to your Desktop
First run may take longer due to PyTorch initialization

## ✨ Features
Downloader
Supports YouTube & SoundCloud
Converts audio to high-quality WAV
Adds track metadata
Smart Sorter
Analyzes audio features (tempo, energy, spectral data)
Classifies into genres/moods like:
House, Techno, Trance
Hip-Hop, Jazz, Classical
Ambient, Lo-Fi, etc.
Automatically organizes files into folders
