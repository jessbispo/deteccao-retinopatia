#!/bin/bash
set -e

rm -rf build dist build_venv

echo "Creating dedicated build environment to optimize dependencies..."
python3 -m venv build_venv
source build_venv/bin/activate

pip install --upgrade pip

pip install pyinstaller streamlit opencv-python-headless Pillow numpy matplotlib

# Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pyinstaller --onefile --noconfirm \
    --name "deteccao-retinopatia" \
    --additional-hooks-dir=./hooks \
    --workpath ./build \
    --distpath ./dist \
    --add-data "../src:src" \
    --add-data "../modelo_retina.pth:." \
    --hiddenimport streamlit \
    --hiddenimport cv2 \
    --hiddenimport torchvision \
    --exclude-module tkinter \
    --noupx \
    --clean \
    run_main.py

echo "Build complete! Executable is in packaging/dist/deteccao-retinopatia"
