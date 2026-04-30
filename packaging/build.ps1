# Build script for Windows (PowerShell)

if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
if (Test-Path build_venv) { Remove-Item -Recurse -Force build_venv }

Write-Host "Creating dedicated build environment..."
python -m venv build_venv
& .\build_venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install pyinstaller streamlit opencv-python-headless Pillow numpy matplotlib

# Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pyinstaller --onefile --noconfirm `
    --name "deteccao-retinopatia" `
    --additional-hooks-dir=./hooks `
    --workpath ./build `
    --distpath ./dist `
    --add-data "..\src;src" `
    --add-data "..\modelo_retina.pth;." `
    --hiddenimport streamlit `
    --hiddenimport cv2 `
    --hiddenimport torchvision `
    --exclude-module tkinter `
    --noupx `
    --clean `
    run_main.py

Write-Host "Build complete! Executable is in packaging\dist\deteccao-retinopatia.exe"
