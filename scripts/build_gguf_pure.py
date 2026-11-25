"""Export model.safetensors to a GGUF that FUTO Keyboard (llama.cpp based) can load.

Three things are required beyond the keyboardlm.* metadata:
  1. Tensor names in GGUF form (token_embd.weight, blk.N.attn_q.weight, ...).
     HF names make llama.cpp fail while loading the model.
  2. Q/K weights permuted for llama.cpp RoPE (adjacent pairs, HF rotates halves).
  3. tokenizer.ggml.* keys (llama.cpp refuses to load a model without a vocabulary).
     FUTO still tokenizes with the embedded SentencePiece proto.
"""
import numpy as np
import sentencepiece as spm
from safetensors import safe_open

import config
import gguf

QK_SUFFIXES = ("attn_q.weight", "attn_k.weight")


def permute_rows_for_rope(weight: np.ndarray, n_head: int) -> np.ndarray:
    """HF stores rotary dims as two halves, llama.cpp as adjacent pairs."""
    rows = weight.shape[0]
    return (
        weight.reshape(n_head, 2, rows // n_head // 2, *weight.shape[1:])
        .swapaxes(1, 2)
        .reshape(weight.shape)
    )


def load_tensors(path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Keine model.safetensors gefunden unter {path}")
    with safe_open(str(path), framework="numpy") as f:
        return {name: f.get_tensor(name).astype(np.float32) for name in f.keys()}


def to_gguf_tensors(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, config.BLOCK_COUNT)
    converted = {}
    for hf_name, weight in tensors.items():
        if hf_name.endswith("rotary_emb.inv_freq"):
            continue
        gguf_name = name_map.get_name(hf_name, try_suffixes=(".weight", ".bias"))
        if gguf_name is None:
            raise KeyError(f"Kein GGUF-Name fuer Tensor {hf_name}")
        if gguf_name.endswith(QK_SUFFIXES):
            weight = permute_rows_for_rope(weight, config.HEAD_COUNT)
        converted[gguf_name] = weight
    if "output.weight" not in converted:
        # Tied embeddings: FUTO's vendored llama.cpp has no fallback and requires output.weight
        converted["output.weight"] = converted["token_embd.weight"]
    return converted


def write_model_metadata(writer: gguf.GGUFWriter) -> None:
    writer.add_string("general.name", config.MODEL_NAME)
    writer.add_string("general.type", config.TYPE)
    writer.add_string("general.size_label", config.SIZE_LABEL)
    writer.add_string("general.description", config.HISTROY)
    writer.add_string("general.author", config.AUTHOR)
    writer.add_string("general.license", config.LICENSE)
    writer.add_context_length(config.CONTEXT_LENGTH)
    writer.add_embedding_length(config.EMBEDDING_LENGTH)
    writer.add_block_count(config.BLOCK_COUNT)
    writer.add_feed_forward_length(config.FEED_FORWARD_LENGTH)
    writer.add_head_count(config.HEAD_COUNT)
    writer.add_head_count_kv(config.HEAD_COUNT_KV)
    writer.add_layer_norm_rms_eps(config.LAYER_NORM_RMS_EPSILON)
    writer.add_key_length(config.KEY_LENGTH)
    writer.add_value_length(config.VALUE_LENGTH)
    writer.add_rope_dimension_count(config.ROPE_DIMENSION_COUNT)
    writer.add_rope_freq_base(10000.0)
    writer.add_vocab_size(config.VOCAB_SIZE)


def write_futo_metadata(writer: gguf.GGUFWriter, spm_bytes: bytes) -> None:
    writer.add_string("keyboardlm.languages", config.LANGUAGE)
    writer.add_uint32("keyboardlm.finetuning_count", config.FINETUNING_COUNT)
    writer.add_string("keyboardlm.history", config.HISTROY)
    writer.add_string("keyboardlm.features", config.FUTO_FEATURES)
    writer.add_string("keyboardlm.ext_tokenizer_type", config.FUTO_TOKENIZER_TYPE)
    # bytes -> ARRAY[UINT8], which is what FUTO's ModelMeta.cpp reads
    writer.add_array("keyboardlm.ext_tokenizer_data", spm_bytes)


def write_llama_vocab(writer: gguf.GGUFWriter, sp: spm.SentencePieceProcessor) -> None:
    if sp.vocab_size() != config.VOCAB_SIZE:
        raise ValueError(
            f"Tokenizer hat {sp.vocab_size()} Pieces, config.VOCAB_SIZE ist {config.VOCAB_SIZE}"
        )
    pieces = range(sp.vocab_size())
    writer.add_tokenizer_model("llama")
    writer.add_token_list([sp.id_to_piece(i).encode("utf-8") for i in pieces])
    writer.add_token_scores([sp.get_score(i) for i in pieces])
    writer.add_token_types([_token_type(sp, i) for i in pieces])
    writer.add_bos_token_id(sp.bos_id())
    writer.add_eos_token_id(sp.eos_id())
    writer.add_unk_token_id(sp.unk_id())
    writer.add_pad_token_id(sp.pad_id())


def _token_type(sp: spm.SentencePieceProcessor, token_id: int) -> int:
    if sp.is_unknown(token_id):
        return gguf.TokenType.UNKNOWN
    if sp.is_control(token_id):
        return gguf.TokenType.CONTROL
    if sp.is_byte(token_id):
        return gguf.TokenType.BYTE
    return gguf.TokenType.NORMAL


def resolve_tokenizer_path():
    path = config.TOKENIZER_PATH
    return path if path.exists() else config.FALLBACK_TOKENIZER_PATH


def convert_to_futo_gguf() -> None:
    print(">>> Starte direkten FUTO-GGUF-Export <<<")
    tensors = to_gguf_tensors(load_tensors(config.SAFETENSOR_PATH))
    print(f"Geladene Tensors: {len(tensors)}")

    tokenizer_path = resolve_tokenizer_path()
    spm_bytes = tokenizer_path.read_bytes()
    sp = spm.SentencePieceProcessor(model_proto=spm_bytes)

    writer = gguf.GGUFWriter(config.RAW_GGUF_NAME, arch="llama")
    write_model_metadata(writer)
    write_futo_metadata(writer, spm_bytes)
    write_llama_vocab(writer, sp)

    for name, weight in tensors.items():
        writer.add_tensor(name, weight)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f">>> FERTIG: {config.RAW_GGUF_NAME} <<<")


if __name__ == "__main__":
    convert_to_futo_gguf()
