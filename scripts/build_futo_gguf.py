import os
import re
import shutil
import subprocess
from pathlib import Path

# --- Pfad-Konfiguration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "futo_de_final"
TOKENIZER_SRC = ROOT_DIR / "tokenizer" / "futo_de_15k.model"
LLAMA_DIR = ROOT_DIR / "llama.cpp"

OUT_F16_GGUF = ROOT_DIR / "futo_de_33m_f16.gguf"
OUT_FUTO_GGUF = ROOT_DIR / "futo_de_33m_futo.gguf"

# 1. llama.cpp klonen falls nicht vorhanden
if not LLAMA_DIR.exists():
    print("[1/5] Klone llama.cpp...")
    subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp", str(LLAMA_DIR)], check=True)

# 2. Sauberen Zustand im llama.cpp-Ordner wiederherstellen
subprocess.run(["git", "-C", str(LLAMA_DIR), "checkout", "."], check=False)

# 3. llama-quantize bei Bedarf via CMake kompilieren
QUANTIZE_BIN = LLAMA_DIR / "build" / "bin" / "llama-quantize"
if not QUANTIZE_BIN.exists():
    alt_bin = LLAMA_DIR / "llama-quantize"
    if alt_bin.exists():
        QUANTIZE_BIN = alt_bin

if not QUANTIZE_BIN.exists():
    print("[2/5] Kompiliere llama-quantize via CMake...")
    BUILD_DIR = LLAMA_DIR / "build"
    subprocess.run(["cmake", "-B", str(BUILD_DIR), "-DGGML_CUDA=OFF", "-DGGML_HIPBLAS=OFF"], cwd=LLAMA_DIR, check=True)
    subprocess.run(["cmake", "--build", str(BUILD_DIR), "--target", "llama-quantize", "-j"], cwd=LLAMA_DIR, check=True)

print(f"[✓] Quantize-Binary bereit: {QUANTIZE_BIN}")

# 4. Tokenizer für den Export bereitstellen
shutil.copy(TOKENIZER_SRC, MODEL_DIR / "tokenizer.model")

# 5. FUTO-Metadaten automatisch lokalisieren und injizieren
print("[3/5] Patche FUTO-Metadaten in llama.cpp...")

# Dynamische Suche nach base.py oder Dateien mit set_gguf_parameters
target_file = None
base_py_candidates = list(LLAMA_DIR.rglob("base.py"))
if base_py_candidates:
    target_file = base_py_candidates[0]
else:
    for py_file in LLAMA_DIR.rglob("*.py"):
        try:
            txt = py_file.read_text(encoding="utf-8", errors="ignore")
            if "def set_gguf_parameters" in txt or "def set_vocab" in txt:
                target_file = py_file
                break
        except Exception:
            continue

if not target_file or not target_file.exists():
    raise FileNotFoundError("Konnte die Model-Basisdatei in llama.cpp nicht finden.")

content = target_file.read_text(encoding="utf-8")

futo_payload = (
    '\n        # --- FUTO KEYBOARD METADATA & TOKENIZER ---\n'
    '        self.gguf_writer.add_string("general.name", "German v1")\n'
    '        self.gguf_writer.add_string("keyboardlm.languages", "de")\n'
    '        self.gguf_writer.add_uint32("keyboardlm.finetuning_count", 0)\n'
    '        self.gguf_writer.add_string("keyboardlm.history", "German model created")\n'
    '        self.gguf_writer.add_string("keyboardlm.features", "base_v1 inverted_space")\n'
    '        self.gguf_writer.add_string("keyboardlm.ext_tokenizer_type", "sentencepiece")\n'
    '        import pathlib\n'
    '        _spm = pathlib.Path(getattr(self, "dir_model", ".")) / "tokenizer.model"\n'
    '        if _spm.exists():\n'
    '            with open(_spm, "rb") as _f:\n'
    '                self.gguf_writer.add_array("keyboardlm.ext_tokenizer_data", list(_f.read()))\n'
    '        # ------------------------------------------\n'
)

# Muster für Funktionsdefinitionen
pattern = r"(def\s+(?:set_gguf_parameters|set_vocab|prepare)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:\n)"
match = re.search(pattern, content)

if not match:
    raise RuntimeError(f"Konnte keinen Funktionsanker in {target_file} finden!")

insert_pos = match.end()
new_content = content[:insert_pos] + futo_payload + content[insert_pos:]
target_file.write_text(new_content, encoding="utf-8")
print(f"[✓] FUTO-Metadaten erfolgreich in {target_file.relative_to(LLAMA_DIR)} injiziert.")

# 6. Konvertierung nach FP16 GGUF
print("[4/5] Konvertiere HuggingFace-Modell nach GGUF (FP16)...")
CONVERT_SCRIPT = LLAMA_DIR / "convert_hf_to_gguf.py"
subprocess.run([
    "python", str(CONVERT_SCRIPT),
    str(MODEL_DIR),
    "--outfile", str(OUT_F16_GGUF),
    "--outtype", "f16"
], check=True)

# 7. Quantisierung auf Q8_0
print("[5/5] Quantisiere zu Q8_0...")
subprocess.run([
    str(QUANTIZE_BIN),
    str(OUT_F16_GGUF),
    str(OUT_FUTO_GGUF),
    "Q8_0"
], check=True)

# 8. Temporäre FP16-Datei löschen
if OUT_F16_GGUF.exists():
    OUT_F16_GGUF.unlink()

print(f"\n[✓ Fertig!] Release-Datei erstellt: {OUT_FUTO_GGUF.name} ({OUT_FUTO_GGUF.stat().st_size / 1024 / 1024:.2f} MB)")