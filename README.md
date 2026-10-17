# FUTO Keyboard German LM (33M)

Ein ultrakompaktes deutsches Sprachmodell (~33M Parameter, LLaMA-Architektur) für latenzfreie, lokale Wortvorhersage und Autokorrektur in der Open-Source-Tastatur **[FUTO Keyboard](https://keyboard.futo.org/)** unter Android.

---

## Features

- **Ultrakompakt:** ~33 Millionen Parameter – optimiert für extrem geringen Akku- und RAM-Verbrauch auf Smartphones.
- **FUTO-kompatibler Tokenizer:** 15.000 Token Custom-SentencePiece-Vokabular mit `treat_whitespace_as_suffix`-Unterstützung.
- **Q8_0 GGUF-Export:** ~35 MB Dateigröße bei INT8 Quantisierung mit nahezu voller FP16 Vorhersagequalität.
- **Vollständig Offline & Privat:** Sämtliche Inferenz läuft zu 100 % lokal auf dem Gerät.
- **Hardwarebeschleunigte Pipeline:** Schnelles Training auf PyTorch mit AMD ROCm oder NVIDIA CUDA.

## Nutzung

**In-App:** 
-> Öffne die "FUTO-Keyboard"-App (oder installiere sie falls noch nicht geschehen)<br>-> Gehe auf "Textvorhersage" und dann auf "Transformer-Modelle"<br>-> Wähle "Import from file" und wähle die .gguf-Datei aus<br>-> Wähle abschließend Deutsch als Sprache für das Model aus (falls Deutsch in der Auswahl nicht auftaucht musst du es unter "Sprachen und Modelle" hinzufügen, hier kannst du auch das Model unter "Transformer-Model" hinzufügen)

## Baue das Transformer-Model selbst
- unter experimental

---

## Projektstruktur

```text
futo-keyboard-german/
├── README.md
├── requirements.txt
├── DataSet/
│   └── (corpus_de_expert.txt) # Korpus - Huggingface 
├── tokenizer/
│   ├── futo_de_15k.model      # SentencePiece Tokenizer
│   └── futo_de_15k.vocab
├── scripts/
│   ├── build_corpus.py     # Datensatzbereinigung & Augmentation
│   ├── train_tokenizer.py  # Tokenizer bauen
│   ├── train_model.py      # PyTorch LLaMA-Training
│   └── build_futo_gguf.py  # llama.cpp GGUF Export & Quantisierung
└── patches/
    └── futo_metadata.patch    # llama.cpp FUTO-Header Patch
