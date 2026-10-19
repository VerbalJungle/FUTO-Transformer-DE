import os
import pathlib
import shutil
import subprocess
import sys
import config

def check_dependencies():
    """Prüft grundlegende CLI-Tools zum Klonen und Kompilieren."""
    for tool in ["git", "cmake"]:
        if not shutil.which(tool):
            raise SystemExit(
                f"Fehler: '{tool}' wurde auf dem System nicht gefunden. "
                f"Bitte installiere es (z. B. 'sudo apt install {tool}')."
            )

def find_or_build_llama_quantize() -> str:
    """Finde ein existierendes llama-quantize oder baue es automatisch."""
    # 1. Bekannte lokale Pfade oder PATH prüfen
    candidates = [
        "./llama.cpp/build/bin/llama-quantize",
        "./llama.cpp/build/bin/quantize",
        "./llama.cpp/llama-quantize",
        "./llama.cpp/quantize",
        "llama-quantize",
        "quantize"
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
        if shutil.which(c):
            return shutil.which(c)

    print(">>> 'llama-quantize' nicht gefunden. Starte Neubau von llama... <<<")
    check_dependencies()

    llama_dir = config.LLAMA_DIR

    # 2. Repository klonen falls noch nicht vorhanden
    if not llama_dir.exists():
        print(">>> Klone llama.cpp Repository... <<<")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(llama_dir)],
            check=True
        )
    else:
        print(">>> llama.cpp Verzeichnis bereits vorhanden. <<<")

    # 3. Kompilieren via CMake
    build_dir = llama_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    print(">>> Konfiguriere CMake... <<<")
    subprocess.run(["cmake", "-B", str(build_dir), "-S", str(llama_dir)], check=True)

    print(">>> Kompiliere llama-quantize (dies kann kurz dauern)... <<<")
    num_cores = str(os.cpu_count() or 4)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "llama-quantize", "-j", num_cores],
        check=True
    )

    # 4. Neu gebautes Binary ermitteln
    built_binary = build_dir / "bin" / "llama-quantize"
    if not built_binary.exists():
        # In B3600 war es noch nur quantize
        built_binary = build_dir / "bin" / "quantize"

    if not (built_binary.exists() and os.access(str(built_binary), os.X_OK)):
        raise FileNotFoundError(f"Kompilierung abgeschlossen, aber kein Binary unter {built_binary} gefunden.")

    print(f">>> Kompilierung erfolgreich: {built_binary.resolve()} <<<\n")
    return str(built_binary.resolve())

def main():
    input_file = config.OUTPUT_GGUF_PATH
    if not os.path.exists(input_file):
        raise SystemExit(
            f"Fehler: Ausgangsmodell {input_file} nicht gefunden. "
            "Bitte erstelle zuerst die Basis-GGUF-Datei."
        )

    quant_tool = find_or_build_llama_quantize()
    print(f">> Verwende Quantisierungs-Tool: {quant_tool}")

    levels = {
        "1": ("Q4_K_M", "German-0_9_1-q4km.gguf"),
        "2": ("Q5_K_M", "German-0_9_1-q5km.gguf"),
        "3": ("Q8_0",   "German-0_9_1-q80.gguf")
    }

    generated_files = {}

    print("\n" + "="*45)
    print(" STARTE QUANTISIERUNG")
    print("="*45)

    for key, (mode, outfile) in levels.items():
        print(f"\n[Stufe {key}] Erstelle {mode} ({outfile})...")
        cmd = [quant_tool, input_file, outfile, mode]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if os.path.exists(outfile):
                size_mb = os.path.getsize(outfile) / (1024 * 1024)
                generated_files[key] = (outfile, size_mb)
                print(f"Erfolgreich! Dateigröße: {size_mb:.2f} MB")
        except subprocess.CalledProcessError as e:
            print(f"Fehler bei {mode}: {e.stderr.strip()}")

    if not generated_files:
        print("\nKeine Modelle konnten quantisiert werden.")
        return

    print("\n" + "="*45)
    print(" ZUSAMMENFASSUNG & AUSWAHL")
    print("="*45)
    for key, (filename, size) in generated_files.items():
        mode_name = levels[key][0]
        print(f" [{key}] {mode_name:7} -> {filename} ({size:.2f} MB)")
    print(" [4] Alle behalten / Nicht überschreiben")
    print("="*45)

    choice = input(f"\nWelche QUantisierung möchtest du als finales Modell nutzen? (1/2/3/4): ").strip()

    if choice in generated_files:
        # Übrige generierte Varianten aufräumen
        for k, (fname, _) in generated_files.items():
            if k != choice and os.path.exists(fname):
                os.remove(fname)

        print(f"\n>>> Fertig! {levels[choice][1]} ist nun die gewählte {levels[choice][0]}-Version. <<<")
    else:
        print("\nKeine Ersetzung vorgenommen. Alle generierten Dateien verbleiben im Verzeichnis.")

if __name__ == "__main__":
    main()