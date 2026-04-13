#!/bin/bash
set -e

rm -rf build dist

python3 -m PyInstaller "Music_Suite.spec"

echo "✅ Build complete: dist/Music Suite.app"