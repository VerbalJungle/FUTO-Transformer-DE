"""Export model.safetensors to a GGUF that FUTO Keyboard (llama.cpp based) can load.

Three things are required beyond the keyboardlm.* metadata:
  1. Tensor names in GGUF form (token_embd.weight, blk.N.attn_q.weight, ...).
     HF names make llama.cpp fail while loading the model.
  2. Q/K weights permuted for llama.cpp RoPE (adjacent pairs, HF rotates halves).
  3. tokenizer.ggml.* keys (llama.cpp refuses to load a model without a vocabulary).
     FUTO still tokenizes with the embedded SentencePiece proto.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import gguf


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
        if gguf_name.endswith("attn_q.weight"):
            weight = permute_rows_for_rope(weight, config.HEAD_COUNT)
        elif gguf_name.endswith("attn_k.weight"):
            weight = permute_rows_for_rope(weight, config.HEAD_COUNT_KV)
        converted[gguf_name] = weight
    if "output.weight" not in converted:
        # Tied embeddings: FUTO's vendored llama.cpp has no fallback and requires output.weight
        converted["output.weight"] = converted["token_embd.weight"]
    return converted


def write_model_metadata(writer: gguf.GGUFWriter, vocab_size: int) -> None:
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
    writer.add_vocab_size(vocab_size)


def write_futo_metadata(writer: gguf.GGUFWriter, spm_bytes: bytes) -> None:
    writer.add_string("keyboardlm.languages", config.LANGUAGE)
    writer.add_uint32("keyboardlm.finetuning_count", config.FINETUNING_COUNT)
    writer.add_string("keyboardlm.history", config.HISTROY)
    writer.add_string("keyboardlm.features", config.FUTO_FEATURES)
    writer.add_string("keyboardlm.ext_tokenizer_type", config.FUTO_TOKENIZER_TYPE)
    # bytes -> ARRAY[UINT8], which is what FUTO's ModelMeta.cpp reads
    writer.add_array("keyboardlm.ext_tokenizer_data", spm_bytes)


def validate_futo_tokenizer(
    sp: spm.SentencePieceProcessor, expected_vocab_size: int | None = config.VOCAB_SIZE
) -> None:
    if expected_vocab_size is not None and sp.vocab_size() != expected_vocab_size:
        raise ValueError(
            f"Tokenizer hat {sp.vocab_size()} Pieces, erwartet wurde {expected_vocab_size}"
        )
    features = set(config.FUTO_FEATURES.split())
    if "xbu_char_autocorrect_v1" in features:
        if sp.unk_id() != config.UNK_TOKEN_ID or sp.bos_id() != config.BOS_TOKEN_ID:
            raise ValueError(
                f"FUTO xbu_char_autocorrect_v1 erfordert unk_id={config.UNK_TOKEN_ID} und "
                f"bos_id={config.BOS_TOKEN_ID} (ist: unk={sp.unk_id()}, bos={sp.bos_id()})"
            )
        unk = sp.unk_id()
        for tok in ("<XBU>", "<XBC>", "<XEC>"):
            if sp.piece_to_id(tok) == unk:
                raise ValueError(f"FUTO Special Token {tok} fehlt im Tokenizer")
        char_a_id = sp.piece_to_id("<CHAR_A>")
        if char_a_id == unk:
            raise ValueError("FUTO Special Token <CHAR_A> fehlt im Tokenizer")
        for i in range(26):
            expected_tok = f"<CHAR_{chr(ord('A') + i)}>"
            actual_id = sp.piece_to_id(expected_tok)
            if actual_id != char_a_id + i:
                raise ValueError(
                    f"FUTO <CHAR_A>..<CHAR_Z> nicht lückenlos: {expected_tok}={actual_id}, "
                    f"erwartet {char_a_id + i}"
                )


def write_llama_vocab(writer: gguf.GGUFWriter, sp: spm.SentencePieceProcessor) -> None:
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
    if sp.is_unused(token_id):
        return gguf.TokenType.UNUSED
    if sp.is_byte(token_id):
        return gguf.TokenType.BYTE
    return gguf.TokenType.NORMAL


def resolve_tokenizer_path() -> Path:
    path = config.TOKENIZER_PATH
    return path if path.exists() else config.FALLBACK_TOKENIZER_PATH


def convert_to_futo_gguf(
    safetensors_path: Path = config.SAFETENSOR_PATH,
    tokenizer_path: Path | None = None,
    output_path: Path = config.OUTPUT_GGUF_PATH,
    allow_custom_vocab: bool = False,
) -> None:
    print(">>> Starte direkten FUTO-GGUF-Export <<<")
    tensors = to_gguf_tensors(load_tensors(safetensors_path))
    print(f"Geladene Tensors: {len(tensors)}")

    tok_path = tokenizer_path if tokenizer_path is not None else resolve_tokenizer_path()
    spm_bytes = tok_path.read_bytes()
    sp = spm.SentencePieceProcessor(model_proto=spm_bytes)
    validate_futo_tokenizer(
        sp, expected_vocab_size=None if allow_custom_vocab else config.VOCAB_SIZE
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output_path), arch="llama")
    write_model_metadata(writer, vocab_size=sp.vocab_size())
    write_futo_metadata(writer, spm_bytes)
    write_llama_vocab(writer, sp)

    for name, weight in tensors.items():
        writer.add_tensor(name, weight)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f">>> FERTIG: {output_path} <<<")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exportiert Safetensors als FUTO-GGUF.")
    parser.add_argument("--safetensors", type=Path, default=config.SAFETENSOR_PATH)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_GGUF_PATH)
    parser.add_argument("--allow-custom-vocab", action="store_true")
    args = parser.parse_args()
    convert_to_futo_gguf(
        safetensors_path=args.safetensors,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        allow_custom_vocab=args.allow_custom_vocab,
    )


if __name__ == "__main__":
    main()
