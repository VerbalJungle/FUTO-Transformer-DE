import sentencepiece as spm

VOCAB_SIZE = 15008  # FUTO-Standardgröße (unter 15 MB Speicher)
CORPUS_FILE = "../DataSet/corpus_de_expert1.txt" # Verwendeter Korpus 
MODEL_PREFIX = "futo_de_15k_1"

print(f"Trainiere SentencePiece BPE-Tokenizer (Vocab: {VOCAB_SIZE})...")

# Training mit Space-Suffix-Spezifikation für Keyboard-Latenz
spm.SentencePieceTrainer.train(
    input=CORPUS_FILE,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=0.9995,
    treat_whitespace_as_suffix=True,  # Entscheidend für Tastatur-Inferenz
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=[
        "<correct>", "</correct>",  # Marker für Tastatur-Autokorrektur
    ]
)

print(f"Tokenizer erfolgreich erstellt: {MODEL_PREFIX}.model und {MODEL_PREFIX}.vocab")
