#!/bin/bash
set -e

echo "=== Erstelle Virtual Environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installiere Abhängigkeiten ==="
pip install --upgrade pip

# Erkennung für AMD GPU / ROCm vs. Standard CPU/CUDA
if command -v rocminfo &> /dev/null; then
    echo "[✓] AMD ROCm erkannt -> Installiere PyTorch ROCm..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
else
    echo "[!] Keine AMD GPU erkannt -> Installiere Standard PyTorch..."
    pip install torch torchvision
fi

pip install -r requirements.txt

echo "=== Fertig! Aktiviere mit: source venv/bin/activate ==="
