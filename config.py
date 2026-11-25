# config.py
from pathlib import Path

# --- PFADE ---
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "DataSet"
KORPUS_NAME = "corpus_de_v2.txt"
CORPUS_PATH = DATASET_DIR / KORPUS_NAME
TOKENIZER_DIR = BASE_DIR / "tokenizer"
TOKENIZER_PREFIX = TOKENIZER_DIR / "futo_de_15k_v2"
MODEL_DIR = BASE_DIR / "futo_de_final"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.model"
FALLBACK_TOKENIZER_PATH = TOKENIZER_DIR / "futo_de_15k_v2.model"
SAFETENSOR_PATH = MODEL_DIR / "model.safetensors"
LLAMA_DIR = BASE_DIR / "llama.cpp"
ORIGINALMODEL = BASE_DIR / "Models" / "ml4_q6_k.gguf"  # bitte entsprechend des Speicherortes anpassen

# GGUF Output-Namen
RAW_GGUF_NAME = "German-0_9_1-unquantized.gguf"
OUTPUT_GGUF_PATH = BASE_DIR / "Models" / RAW_GGUF_NAME

# --- MODELL-METADATEN & ARCHITEKTUR  ---
MODEL_NAME = "Futo_De_0_9_1"
SIZE_LABEL = "33M"
TYPE = "model"
LANGUAGE = "de"
AUTHOR = "VerbalJungle"
LICENSE = "MIT"

BLOCK_COUNT = 8
CONTEXT_LENGTH = 2048
EMBEDDING_LENGTH = 512
FEED_FORWARD_LENGTH = 1376
HEAD_COUNT = 8
HEAD_COUNT_KV = 8
LAYER_NORM_RMS_EPSILON = 1e-06
KEY_LENGTH = 64
VALUE_LENGTH = 64
VOCAB_SIZE = 15008
ROPE_DIMENSION_COUNT = 64
TIE_WORD_EMBEDDINGS = False

# FUTO C++ (LanguageModel.cpp) hardcodet BOS=1 und erwartet fehlende Token als 0 (UNK)
UNK_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 3
FUTO_SPECIAL_TOKENS = [
    "<XBU>",
    "<XBC>",
    "<XEC>",
    *[f"<CHAR_{chr(c)}>" for c in range(ord("A"), ord("Z") + 1)],
]

# --- FUTO SPEZIFISCH KEYBOARDLM ---
FINETUNING_COUNT = 0
HISTROY = "German custom prediction model for FUTO keyboard"
FUTO_FEATURES = "base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1"
FUTO_TOKENIZER_TYPE = "sentencepiece"

# --- QUANTISIERUNG ---
QUANT_LEVELS = {
    "1": ("Q4_K_M", "transformer_de_DE_q4km.gguf"),
    "2": ("Q5_K_M", "transformer_de_DE_q5km.gguf"),
    "3": ("Q8_0",   "transformer_de_DE_q80.gguf"),
}
