# FUTO Keyboard German LM (40M)

Ein ultrakompaktes deutsches Sprachmodell (~40M Parameter, LLaMA-Architektur mit separatem Output-Head) für latenzfreie, lokale Wortvorhersage, Wortvervollständigung und Autokorrektur in der Open-Source-Tastatur **[FUTO Keyboard](https://keyboard.futo.org/)** unter Android.

---

## Features

- **Ultrakompakt & Offline:** ~40 Millionen Parameter (8 Layer, 512 Hidden Size) – läuft zu 100 % lokal auf dem Gerät ohne Internetverbindung.
- **Next-Word-Prediction (`base_v1`, `inverted_space`):** Sagt passende nächste Wörter aus dem Satzkontext vorher.
- **Live-Wortvervollständigung & Tippfehlerkorrektur (`xbu_char_autocorrect_v1`):** Vervollständigt angefangene Wörter schon nach wenigen Buchstaben und korrigiert Tippfehler, Buchstabendreher und ausgelassene Zeichen während des Tippens.
- **Touch-Koordinaten-Mixing (`char_embed_mixing_v1`):** Verarbeitet gewichtete Tasten-Wahrscheinlichkeiten direkt aus der FUTO-Tastatur.
- **Deutsche Umlaute & Rechtschreibung:** Automatische Wiederherstellung von `ä`, `ö`, `ü`, `ß` sowie kontextabhängige Groß-/Kleinschreibung *(ohne Swipe-Typing / `xc0_swipe_typing_v1`)*.
- **FUTO-kompatibler Tokenizer:** 15.008 Token SentencePiece-BPE-Vokabular mit `treat_whitespace_as_suffix`, Byte-Fallback, `split_digits` und den FUTO-Spezialtokens (`<XBU>`, `<XBC>`, `<XEC>`, `<CHAR_A>`–`<CHAR_Z>`).
- **Quantisierte GGUF-Varianten:**
  - `German-v1-q5km.gguf` (**~30 MB** – **Empfohlen** als Standard für die meisten Geräte)
  - `German-v1-q80.gguf` (**~42 MB** – maximale Präzision und hohe Geschwindigkeit)
  - `German-v1-q4km.gguf` (**~27 MB** – für Geräte mit sehr knappem Speicher)
  - *(zum Vergleich: unquantisiert `F32` ~156 MB)*

---

## Nutzung

**Welches Modell wählen?**  
Für normale Nutzer empfiehlt sich **`German-v1-q5km.gguf`**. Wer genügend Speicher hat und maximale Genauigkeit möchte, greift zu **`German-v1-q80.gguf`**.

**In-App-Installation (FUTO Keyboard):**
1. Öffne die **FUTO Keyboard**-App.
2. Gehe zu **Textvorhersage** → **Transformer-Modelle**.
3. Wähle **Import from file** und lade die gewünschte `.gguf`-Datei aus `Models/` (oder dem GitHub-Release).
4. Weise das Modell unter **Sprachen und Modelle** der Sprache **Deutsch** zu (falls Deutsch noch nicht aktiv ist, zuerst dort hinzufügen).

---

## Baue das Transformer-Modell selbst

Nahezu alle Pfade, Architektur- und Metadaten-Parameter lassen sich zentral in `config.py` anpassen.  
Die Rohdaten für Reddit(r/fragReddit) sind unter Artic Shift im jsonl-Format verfügbar, für Wikipedia bieten sich die 1M-Sentence Datensätze vom Leipzig Corpora Collection an

### 1. Umgebung einrichten
```bash
./setup.sh
source venv/bin/activate
```

### 2. Pipeline ausführen
```bash
# 1. Korpus bauen (~85 % FragReddit + ~15 % Wikipedia 2021 -> DataSet/corpus_de_v2.txt)
python scripts/build_corpus.py

# 2. FUTO-kompatiblen SentencePiece-Tokenizer trainieren & validieren
python scripts/train_tokenizer.py

# 3. Modell trainieren (Sequence Packing + <XBU>-Augmentierung + Eval-Split)
#    Hinweis: benötigt GPU (ROCm/CUDA); für einen schnellen Testlauf --dry-run nutzen
python scripts/train_model.py

# 4. Als FUTO-kompatible GGUF-Datei nach Models/ exportieren
python scripts/build_gguf_pure.py

# 5. In Q4_K_M / Q5_K_M / Q8_0 quantisieren
python scripts/quantize_model.py
```

### 3. Benchmark (optional)
`Benchmark.py` testet Next-Word-Prediction und `<XBU>`-Autokorrektur direkt über `llama.cpp` (`libllama.so`) oder reines PyTorch (ohne `llama-cpp-python`):
```bash
# Standard: testet Models/transformer_de_DE_q80.gguf
python scripts/Testing/Benchmark.py

# Alle GGUFs in Models/ + PyTorch-Referenz (futo_de_final/) vergleichen
python scripts/Testing/Benchmark.py --all --include-pt --summary-only
```

---

## Projektstruktur

```text
futo-keyboard-german/
├── README.md
├── LICENSE.md
├── requirements.txt
├── setup.sh
├── config.py                     # Zentrale Konfiguration (Pfade, Architektur, FUTO-Flags)
├── DataSet/                      # Rohdaten & generierter Korpus (corpus_de_v2.txt)
├── tokenizer/
│   ├── futo_de_15k_v2.model      # Aktiver FUTO-SentencePiece-Tokenizer
│   └── futo_de_15k_v2.vocab
├── scripts/
│   ├── build_corpus.py           # Korpus-Bereinigung, Filterung & Mix
│   ├── train_tokenizer.py        # SentencePiece-Training & FUTO-Validierung
│   ├── train_model.py            # PyTorch LLaMA-Training (Packing & <XBU>-Augmentierung)
│   ├── build_gguf_pure.py        # Pure-Python GGUF-Export inkl. FUTO-Metadaten
│   ├── quantize_model.py         # Quantisierung über llama-quantize
│   └── Testing/
│       ├── Benchmark.py          # Next-Word- & <XBU>-Benchmark (libllama.so & PyTorch)
│       └── compare_to_original.py
├── Models/                       # Exportierte & quantisierte GGUF-Modelle
└── patches/
    └── futo_metadata.patch       # Veralteter llama.cpp FUTO-Header-Patch
```
