import pathlib
from safetensors.torch import load_file
import config
import gguf


def convert_to_futo_gguf():
    model_dir = config.MODEL_DIR
    output_file = config.RAW_GGUF_NAME

    print(">>> Starte direkten FUTO-GGUF-Export <<<")

    # 1. Safetensors laden
    safetensors_path = config.SAFETENSOR_PATH
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Keine model.safetensors gefunden unter {safetensors_path}")

    tensors = load_file(str(safetensors_path))
    print(f"Geladene Tensors: {len(tensors)}")

    # 2. GGUF-Writer initialisieren
    writer = gguf.GGUFWriter(output_file, arch="llama")

    # --- FUTO METADATEN GENERAL ---
    writer.add_string("general.name", config.MODEL_NAME)
    writer.add_string("general.type", config.TYPE)
    writer.add_string("general.size_label", config.SIZE_LABEL)
    writer.add_string("general.describtion", config.HISTROY)
    writer.add_string("general.author", config.AUTHOR)
    writer.add_string("general.license", config.LICENSE)
    writer.add_string("keyboardlm.languages", config.LANGUAGE)

    writer.add_uint32("llama.block_count", config.BLOCK_COUNT)
    writer.add_uint32("llama.context_length", config.CONTEXT_LENGTH)
    writer.add_uint32("llama.embedding_length", config.EMBEDDING_LENGTH)
    writer.add_uint32("llama.feed_forward_length", config.FEED_FORWARD_LENGTH)
    writer.add_uint32("llama.attention.head_count", config.HEAD_COUNT)
    writer.add_uint32("llama.attention.head_count_kv", config.HEAD_COUNT_KV)
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", config.LAYER_NORM_RMS_EPSILON)
    writer.add_uint32("llama.attention.key_length", config.KEY_LENGTH)
    writer.add_uint32("llama.attention.value_length", config.VALUE_LENGTH)
    writer.add_uint32("llama.vocab_size", config.VOCAB_SIZE)
    writer.add_uint32("llama.rope.dimension_count", config.ROPE_DIMENSION_COUNT)

    # --- FUTO SPEZIFISCH KEYBOARDLM---
    writer.add_uint32("keyboardlm.finetuning_count", config.FINETUNING_COUNT)
    writer.add_string("keyboardlm.history", config.HISTROY)
    writer.add_string("keyboardlm.features", config.FUTO_FEATURES)
    writer.add_string("keyboardlm.ext_tokenizer_type", config.FUTO_TOKENIZER_TYPE)

    # 3. Tokenizer direkt als rohe Bytes übergeben (löst automatisch UINT8 aus!)
    tokenizer_path = config.TOKENIZER_PATH
    if not tokenizer_path.exists():
        tokenizer_path = config.FALLBACK_TOKENIZER_PATH

    if tokenizer_path.exists():
        with open(tokenizer_path, "rb") as f:
            token_bytes = f.read()  # Gibt als bytes zurück

        # Übergabe als bytes -> greift sofort die isinstance(val, bytes) Logik!
        writer.add_array("keyboardlm.ext_tokenizer_data", token_bytes)
        print(f">>> Tokenizer ({len(token_bytes):,} Bytes) direkt als UINT8-Bytes eingebettet! <<<")
    else:
        print(">>> Warnung: Kein Tokenizer gefunden! <<<")

    # 4. Gewichte schreiben
    print("Schreibe Tensors...")
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor.detach().numpy())

    # 5. Datei abspeichern
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"\n>>> FERTIG! Saubere FUTO-GGUF-Datei gespeichert unter: {output_file} <<<")


if __name__ == "__main__":
    convert_to_futo_gguf()