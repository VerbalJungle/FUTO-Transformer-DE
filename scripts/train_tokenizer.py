"""Trainiert und validiert den FUTO-kompatiblen SentencePiece-BPE-Tokenizer."""
import argparse
import sys
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def verify_futo_tokenizer(model_path: Path, expected_vocab_size: int | None = None) -> None:
    """Prüft alle harten FUTO-Keyboard-Invarianzbedingungen für SentencePiece."""
    sp = spm.SentencePieceProcessor(model_file=str(model_path))

    if expected_vocab_size is not None and sp.vocab_size() != expected_vocab_size:
        raise AssertionError(
            f"vocab_size ist {sp.vocab_size()}, erwartet {expected_vocab_size}"
        )
    if sp.unk_id() != config.UNK_TOKEN_ID:
        raise AssertionError(f"unk_id={sp.unk_id()}, erwartet {config.UNK_TOKEN_ID}")
    if sp.bos_id() != config.BOS_TOKEN_ID:
        raise AssertionError(f"bos_id={sp.bos_id()}, erwartet {config.BOS_TOKEN_ID}")
    if sp.eos_id() != config.EOS_TOKEN_ID:
        raise AssertionError(f"eos_id={sp.eos_id()}, erwartet {config.EOS_TOKEN_ID}")
    if sp.pad_id() != config.PAD_TOKEN_ID:
        raise AssertionError(f"pad_id={sp.pad_id()}, erwartet {config.PAD_TOKEN_ID}")

    for tok in ("<XBU>", "<XBC>", "<XEC>", "▁"):
        tid = sp.piece_to_id(tok)
        if tid == sp.unk_id():
            raise AssertionError(f"Pflicht-Token '{tok}' fehlt (auf UNK gemappt)")

    char_a_id = sp.piece_to_id("<CHAR_A>")
    if char_a_id == sp.unk_id():
        raise AssertionError("'<CHAR_A>' fehlt")
    for i in range(26):
        tok = f"<CHAR_{chr(ord('A') + i)}>"
        if sp.piece_to_id(tok) != char_a_id + i:
            raise AssertionError(
                f"{tok} hat ID {sp.piece_to_id(tok)}, erwartet {char_a_id + i}"
            )

    byte_count = sum(1 for i in range(sp.vocab_size()) if sp.is_byte(i))
    if byte_count != 256:
        raise AssertionError(f"Erwartet 256 Byte-Tokens, gefunden: {byte_count}")

    pieces = sp.encode_as_pieces("Guten ")
    if pieces != ["Guten▁"] and (not pieces or not pieces[-1].endswith("▁") or pieces[-1] == "▁"):
        raise AssertionError(
            f"Trailing Space nicht als Wort-Suffix erhalten: {pieces!r}"
        )
    print(
        f"[✓] Tokenizer verifiziert: vocab={sp.vocab_size()}, "
        f"<CHAR_A>..Z={char_a_id}..{char_a_id + 25}, bytes={byte_count}"
    )


def train_tokenizer(
    input_file: Path = config.CORPUS_PATH,
    model_prefix: Path = config.TOKENIZER_PREFIX,
    vocab_size: int = config.VOCAB_SIZE,
    input_sentence_size: int = 2_000_000,
) -> Path:
    if not input_file.exists():
        raise FileNotFoundError(f"Korpus nicht gefunden: {input_file}")

    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    print(f"Trainiere SentencePiece BPE-Tokenizer (Vocab: {vocab_size}) -> {model_prefix}")

    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        treat_whitespace_as_suffix=True,
        remove_extra_whitespaces=True,
        byte_fallback=True,
        split_digits=True,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        unk_id=config.UNK_TOKEN_ID,
        bos_id=config.BOS_TOKEN_ID,
        eos_id=config.EOS_TOKEN_ID,
        pad_id=config.PAD_TOKEN_ID,
        user_defined_symbols=config.FUTO_SPECIAL_TOKENS,
    )

    model_path = Path(f"{model_prefix}.model")
    verify_futo_tokenizer(model_path, expected_vocab_size=vocab_size)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainiert den FUTO-DE SentencePiece-Tokenizer.")
    parser.add_argument("--input", type=Path, default=config.CORPUS_PATH)
    parser.add_argument("--model-prefix", type=Path, default=config.TOKENIZER_PREFIX)
    parser.add_argument("--vocab-size", type=int, default=config.VOCAB_SIZE)
    parser.add_argument("--input-sentence-size", type=int, default=2_000_000)
    args = parser.parse_args()
    train_tokenizer(
        input_file=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        input_sentence_size=args.input_sentence_size,
    )


if __name__ == "__main__":
    main()
