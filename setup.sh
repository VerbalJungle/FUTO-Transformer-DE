#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Erstelle Virtual Environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installiere Abhängigkeiten ==="
pip install --upgrade pip

ROCM_INDEX="${ROCM_INDEX:-rocm6.1}"

# Erkennung für AMD GPU / ROCm vs. Standard CPU/CUDA
if command -v rocminfo &> /dev/null; then
    echo "[✓] AMD ROCm erkannt -> Installiere PyTorch ($ROCM_INDEX)..."
    pip install torch --index-url "https://download.pytorch.org/whl/${ROCM_INDEX}"
else
    echo "[!] Keine AMD GPU erkannt -> Installiere Standard PyTorch..."
    pip install torch
fi

pip install -r requirements.txt

for tool in git cmake; do
    if ! command -v "$tool" &> /dev/null; then
        echo "[!] Hinweis: '$tool' fehlt (wird für quantize_model.py / llama.cpp benötigt)."
    fi
done

echo "=== Fertig! Aktiviere mit: source venv/bin/activate ==="
