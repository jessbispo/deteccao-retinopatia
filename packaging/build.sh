#!/bin/bash

# Cleanup previous builds
rm -rf build dist

# Activate virtual environment if it exists
if [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

# Install dependencies if not present
pip install pyinstaller opencv-python-headless torchvision

# Run PyInstaller
pyinstaller --onefile \
    --name "deteccao-retinopatia" \
    --additional-hooks-dir=./hooks \
    --workpath ./build \
    --distpath ./dist \
    --add-data "../src:src" \
    --add-data "../modelo_retina.pth:." \
    --hiddenimport streamlit \
    --hiddenimport cv2 \
    --hiddenimport torchvision \
    --clean \
    run_main.py

echo "Build complete! Executable is in packaging/dist/deteccao-retinopatia"
