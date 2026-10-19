# config.py
from pathlib import Path

# --- PFADE ---
BASE_DIR = Path(__file__).resolve().parent
KORPUS_NAME = "corpus_de_expert.txt"
MODEL_DIR = BASE_DIR / "futo_de_final"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.model"
FALLBACK_TOKENIZER_PATH = BASE_DIR / "tokenizer" / "futo_de_15k.model"
SAFETENSOR_PATH = BASE_DIR / "futo_de_final" / "model.safetensors"
LLAMA_DIR = BASE_DIR / "llama.cpp"
ORIGINALMODEL = BASE_DIR / "Models" / "ml4_q6_k.gguf" # bitte entsprechend des Speicherortes anpassen

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

# --- FUTO SPEZIFISCH KEYBOARDLM ---
FINETUNING_COUNT = 0
HISTROY = "German custom prediction model for FUTO keyboard"
FUTO_FEATURES = "base_v1 inverted_space xbu_char_autocorrect_v1 xc0_swipe_typing_v1 char_embedding_mixing_v1"
FUTO_TOKENIZER_TYPE = "sentencepiece"

# --- QUANTISIERUNG ---
QUANT_LEVELS = {
    "1": ("Q4_K_M", "transformer_de_DE_q4km.gguf"),
    "2": ("Q5_K_M", "transformer_de_DE_q5km.gguf"),
    "3": ("Q8_0",   "transformer_de_DE_q80.gguf"),
}
