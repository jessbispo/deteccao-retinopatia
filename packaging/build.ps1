# Build script for Windows (PowerShell)

# Cleanup previous builds
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist) { Remove-Item -Recurse -Force dist }

# Activate virtual environment if it exists
if (Test-Path ..\.venv\Scripts\Activate.ps1) {
    & ..\.venv\Scripts\Activate.ps1
}

# Install dependencies if not present
pip install pyinstaller opencv-python-headless torchvision

# Run PyInstaller
pyinstaller --onefile `
    --name "deteccao-retinopatia" `
    --additional-hooks-dir=./hooks `
    --workpath ./build `
    --distpath ./dist `
    --add-data "..\src;src" `
    --add-data "..\modelo_retina.pth;." `
    --hiddenimport streamlit `
    --hiddenimport cv2 `
    --hiddenimport torchvision `
    --clean `
    run_main.py

Write-Host "Build complete! Executable is in packaging\dist\deteccao-retinopatia.exe"
