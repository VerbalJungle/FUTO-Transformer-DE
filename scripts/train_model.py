"""Trainiert das FUTO-DE Sprachmodell mit XBU-Autokorrektur-Augmentierung und Sequence Packing."""
import argparse
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

WORD_SPAN_RE = re.compile(r"\S+\s*")
ALPHA_WORD_RE = re.compile(r"^[A-Za-zÄÖÜäöüß]{2,}$")

# Nur Buchstaben auf mobiler QWERTZ-Tastatur (FUTO mappt nur a-z/A-Z auf <CHAR_*>)
QWERTZ_NEIGHBORS = {
    "a": "qwsy", "b": "vghn", "c": "xdfv", "d": "serfxc", "e": "wsdr",
    "f": "drtgcv", "g": "ftzhvb", "h": "gzujbn", "i": "ujko", "j": "huiknm",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxy", "t": "rfgz",
    "u": "zhji", "v": "cfgb", "w": "qase", "x": "ysdc", "y": "asx",
    "z": "tghu",
}
ASCII_LETTERS = "abcdefghijklmnopqrstuvwxyz"


class FutoAugmenter:
    """Erzeugt FUTO-<XBU>-Autokorrektur-Sequenzen und Loss-Masken."""

    def __init__(self, sp: spm.SentencePieceProcessor):
        self.sp = sp
        self.unk_id = sp.unk_id()
        self.bos_id = sp.bos_id()
        self.eos_id = sp.eos_id()
        self.xbu_id = sp.piece_to_id("<XBU>")
        self.xbc_id = sp.piece_to_id("<XBC>")
        self.xec_id = sp.piece_to_id("<XEC>")
        self.char_a_id = sp.piece_to_id("<CHAR_A>")

        if self.unk_id != config.UNK_TOKEN_ID or self.bos_id != config.BOS_TOKEN_ID:
            raise ValueError(
                f"Tokenizer hat unk={self.unk_id}, bos={self.bos_id}, "
                f"FUTO erwartet unk={config.UNK_TOKEN_ID}, bos={config.BOS_TOKEN_ID}"
            )
        for name, tid in (("<XBU>", self.xbu_id), ("<XBC>", self.xbc_id), ("<XEC>", self.xec_id)):
            if tid == self.unk_id:
                raise ValueError(f"Tokenizer fehlt FUTO-Token {name}")
        if self.char_a_id == self.unk_id:
            raise ValueError("Tokenizer fehlt <CHAR_A>")
        for i in range(26):
            tok = f"<CHAR_{chr(ord('A') + i)}>"
            if sp.piece_to_id(tok) != self.char_a_id + i:
                raise ValueError(f"Tokenizer-Reihenfolge fehlerhaft bei {tok}")

    @staticmethod
    def word_to_ascii_letters(word: str, rng: random.Random) -> str:
        """Simuliert FUTOs ASCII-Tastenstrom inkl. Umlaut-Abbildungen."""
        r = rng.random()
        if r < 0.55:
            repl = {"ä": "a", "ö": "o", "ü": "u", "ß": "ss"}
        elif r < 0.75:
            repl = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "s"}
        else:
            # FUTO C++ überspringt non-ASCII-Zeichen im Compose-Buffer
            repl = {"ä": "", "ö": "", "ü": "", "ß": ""}
        out = []
        for ch in word.lower():
            mapped = repl.get(ch, ch)
            for m in mapped:
                if "a" <= m <= "z":
                    out.append(m)
        return "".join(out)

    @staticmethod
    def _single_typo(chars: list[str], rng: random.Random) -> list[str]:
        op = rng.random()
        if len(chars) >= 2 and op < 0.45:
            idx = rng.randrange(len(chars))
            neighbors = QWERTZ_NEIGHBORS.get(chars[idx], "")
            if neighbors:
                chars[idx] = rng.choice(neighbors)
        elif len(chars) >= 2 and op < 0.65:
            idx = rng.randrange(len(chars) - 1)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif len(chars) >= 3 and op < 0.85:
            doubles = [i for i in range(len(chars) - 1) if chars[i] == chars[i + 1]]
            idx = rng.choice(doubles) if doubles else rng.randrange(1, len(chars))
            chars.pop(idx)
        else:
            idx = rng.randrange(len(chars))
            base = chars[idx]
            ins = rng.choice(QWERTZ_NEIGHBORS.get(base, ASCII_LETTERS))
            chars.insert(idx, ins)
        return chars

    def word_to_keystrokes(self, word: str, rng: random.Random) -> list[int]:
        letters = self.word_to_ascii_letters(word, rng)
        if not letters:
            return []

        mode = rng.random()
        if mode < 0.35:
            typed = letters
        elif mode < 0.60 and len(letters) >= 4:
            typed = letters[: rng.randint(2, len(letters) - 1)]
        else:
            chars = list(letters)
            chars = self._single_typo(chars, rng)
            if len(chars) >= 5 and rng.random() < 0.25:
                chars = self._single_typo(chars, rng)
            typed = "".join(chars) or letters

        return [self.char_a_id + (ord(c) - ord("a")) for c in typed if "a" <= c <= "z"]

    def encode_line(
        self, text: str, rng: random.Random, xbu_prob: float
    ) -> tuple[list[int], list[int]]:
        """Gibt (token_ids, loss_mask) mit maskiertem BOS/<XBU>/<CHAR_*>/<XBC> zurück."""
        if xbu_prob <= 0.0 or rng.random() >= xbu_prob:
            ids = [self.bos_id] + self.sp.encode_as_ids(text) + [self.eos_id]
            mask = [0] + [1] * (len(ids) - 1)
            return ids, mask

        spans = list(WORD_SPAN_RE.finditer(text))
        candidates = [
            idx
            for idx, m in enumerate(spans)
            if ALPHA_WORD_RE.match(m.group(0).rstrip(" \t"))
        ]
        if not candidates:
            ids = [self.bos_id] + self.sp.encode_as_ids(text) + [self.eos_id]
            mask = [0] + [1] * (len(ids) - 1)
            return ids, mask

        chosen_idx = rng.choice(candidates)
        ids: list[int] = [self.bos_id]
        mask: list[int] = [0]

        prefix = text[: spans[chosen_idx].start()]
        if prefix:
            p_ids = self.sp.encode_as_ids(prefix)
            ids.extend(p_ids)
            mask.extend([1] * len(p_ids))

        target_span = spans[chosen_idx].group(0)
        core_word = target_span.rstrip(" \t")
        char_ids = self.word_to_keystrokes(core_word, rng)
        t_ids = self.sp.encode_as_ids(target_span)

        if char_ids and t_ids:
            ids.append(self.xbu_id)
            mask.append(0)
            ids.extend(char_ids)
            mask.extend([0] * len(char_ids))
            ids.append(self.xbc_id)
            mask.append(0)
            ids.extend(t_ids)
            mask.extend([1] * len(t_ids))
            ids.append(self.xec_id)
            mask.append(1)
        else:
            ids.extend(t_ids)
            mask.extend([1] * len(t_ids))

        suffix = text[spans[chosen_idx].end() :]
        if suffix:
            s_ids = self.sp.encode_as_ids(suffix)
            ids.extend(s_ids)
            mask.extend([1] * len(s_ids))

        ids.append(self.eos_id)
        mask.append(1)
        return ids, mask


class _PackStreamWriter:
    """Packt variable Sequenzen ohne Padding in feste Blöcke auf Disk."""

    def __init__(self, prefix: Path, block_size: int):
        self.block_size = block_size
        self.tok_path = Path(f"{prefix}.tokens.bin")
        self.mask_path = Path(f"{prefix}.masks.bin")
        self._tok_f = self.tok_path.open("wb")
        self._mask_f = self.mask_path.open("wb")
        self._tok_buf: list[int] = []
        self._mask_buf: list[int] = []
        self.n_blocks = 0

    def add(self, ids: list[int], mask: list[int]) -> None:
        self._tok_buf.extend(ids)
        self._mask_buf.extend(mask)
        while len(self._tok_buf) >= self.block_size:
            tok_arr = np.asarray(self._tok_buf[: self.block_size], dtype=np.uint16)
            mask_arr = np.asarray(self._mask_buf[: self.block_size], dtype=np.uint8)
            self._tok_f.write(tok_arr.tobytes())
            self._mask_f.write(mask_arr.tobytes())
            del self._tok_buf[: self.block_size]
            del self._mask_buf[: self.block_size]
            self.n_blocks += 1

    def close(self) -> None:
        self._tok_f.close()
        self._mask_f.close()


class PackedBinaryDataset(Dataset):
    """Memmap-Dataset für gepackte (input_ids, labels)-Blöcke."""

    def __init__(self, prefix: Path, block_size: int, n_blocks: int):
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.tokens = np.memmap(
            f"{prefix}.tokens.bin", dtype=np.uint16, mode="r", shape=(n_blocks, block_size)
        )
        self.masks = np.memmap(
            f"{prefix}.masks.bin", dtype=np.uint8, mode="r", shape=(n_blocks, block_size)
        )

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids = torch.from_numpy(self.tokens[idx].astype(np.int64))
        mask = torch.from_numpy(self.masks[idx].astype(np.bool_))
        labels = input_ids.clone()
        labels[~mask] = -100
        return {"input_ids": input_ids, "labels": labels}


def prepare_packed_datasets(
    corpus_path: Path,
    sp: spm.SentencePieceProcessor,
    cache_dir: Path,
    block_size: int,
    xbu_prob: float,
    max_lines: int,
    max_val_blocks: int = 20_000,
) -> tuple[PackedBinaryDataset, PackedBinaryDataset]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    train_prefix = cache_dir / "train"
    val_prefix = cache_dir / "val"

    augmenter = FutoAugmenter(sp)
    rng = random.Random(42)
    train_writer = _PackStreamWriter(train_prefix, block_size)
    val_writer = _PackStreamWriter(val_prefix, block_size)

    with corpus_path.open("r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1):
            if max_lines > 0 and line_idx > max_lines:
                break
            text = raw.strip()
            if not text:
                continue
            ids, mask = augmenter.encode_line(text, rng, xbu_prob=xbu_prob)
            if line_idx % 100 == 0 and val_writer.n_blocks < max_val_blocks:
                val_writer.add(ids, mask)
            else:
                train_writer.add(ids, mask)

    train_writer.close()
    val_writer.close()

    # Fallback für sehr kleine Testläufe: mindestens 1 Val-Block abspalten
    if val_writer.n_blocks == 0 and train_writer.n_blocks >= 2:
        last_tok = np.memmap(
            train_writer.tok_path, dtype=np.uint16, mode="r+",
            shape=(train_writer.n_blocks, block_size),
        )[-1].copy()
        last_mask = np.memmap(
            train_writer.mask_path, dtype=np.uint8, mode="r+",
            shape=(train_writer.n_blocks, block_size),
        )[-1].copy()
        val_writer.tok_path.write_bytes(last_tok.tobytes())
        val_writer.mask_path.write_bytes(last_mask.tobytes())
        val_writer.n_blocks = 1
        train_writer.n_blocks -= 1
        os.truncate(train_writer.tok_path, train_writer.n_blocks * block_size * 2)
        os.truncate(train_writer.mask_path, train_writer.n_blocks * block_size)

    if train_writer.n_blocks == 0 or val_writer.n_blocks == 0:
        raise RuntimeError(
            f"Zu wenig Daten für Sequence Packing (train={train_writer.n_blocks}, val={val_writer.n_blocks})"
        )

    meta = {
        "block_size": block_size,
        "n_train": train_writer.n_blocks,
        "n_val": val_writer.n_blocks,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[✓] Gepackte Blöcke (Länge {block_size}): "
        f"Train={train_writer.n_blocks:,} ({train_writer.n_blocks * block_size:,} Tokens), "
        f"Val={val_writer.n_blocks:,}"
    )
    return (
        PackedBinaryDataset(train_prefix, block_size, train_writer.n_blocks),
        PackedBinaryDataset(val_prefix, block_size, val_writer.n_blocks),
    )


def train(args: argparse.Namespace) -> None:
    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if not args.dry_run and not has_cuda:
        raise RuntimeError("FEHLER: ROCm/CUDA erkennt keine GPU! Training wird nicht auf CPU gestartet.")

    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU (dry-run)"
    print(f"Aktives Gerät: {device_name}")

    if not args.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer nicht gefunden: {args.tokenizer}")
    if not args.corpus.exists():
        raise FileNotFoundError(f"Korpus nicht gefunden: {args.corpus}")

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if not args.dry_run and sp.vocab_size() != config.VOCAB_SIZE:
        raise ValueError(
            f"Tokenizer vocab_size={sp.vocab_size()} stimmt nicht mit config.VOCAB_SIZE={config.VOCAB_SIZE} überein"
        )

    output_dir = args.output_dir or (
        Path("/tmp/futo_dry_run_model") if args.dry_run else config.MODEL_DIR
    )
    cache_dir = args.cache_dir or (
        Path("/tmp/futo_dry_run_cache") if args.dry_run else (config.BASE_DIR / "futo_de_cache")
    )
    ckpt_dir = output_dir.parent / f"{output_dir.name}_checkpoints"

    max_lines = args.max_lines if args.max_lines > 0 else (500 if args.dry_run else 0)
    train_ds, val_ds = prepare_packed_datasets(
        corpus_path=args.corpus,
        sp=sp,
        cache_dir=cache_dir,
        block_size=args.block_size,
        xbu_prob=args.xbu_prob,
        max_lines=max_lines,
    )

    model_cfg = LlamaConfig(
        vocab_size=sp.vocab_size(),
        hidden_size=config.EMBEDDING_LENGTH,
        intermediate_size=config.FEED_FORWARD_LENGTH,
        num_hidden_layers=config.BLOCK_COUNT,
        num_attention_heads=config.HEAD_COUNT,
        num_key_value_heads=config.HEAD_COUNT_KV,
        max_position_embeddings=config.CONTEXT_LENGTH,
        rms_norm_eps=config.LAYER_NORM_RMS_EPSILON,
        unk_token_id=config.UNK_TOKEN_ID,
        bos_token_id=config.BOS_TOKEN_ID,
        eos_token_id=config.EOS_TOKEN_ID,
        pad_token_id=config.PAD_TOKEN_ID,
        tie_word_embeddings=config.TIE_WORD_EMBEDDINGS,
    )
    model = LlamaForCausalLM(model_cfg)

    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not use_bf16

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=1 if args.dry_run else args.epochs,
        max_steps=2 if args.dry_run else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0 if args.dry_run else 0.05,
        weight_decay=0.05,
        max_grad_norm=1.0,
        logging_steps=1 if args.dry_run else 100,
        eval_strategy="steps",
        eval_steps=1 if args.dry_run else 500,
        save_strategy="steps",
        save_steps=2 if args.dry_run else 1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=use_bf16,
        fp16=use_fp16,
        use_cpu=not has_cuda,
        dataloader_num_workers=0 if args.dry_run else 4,
        dataloader_pin_memory=has_cuda,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print(f"Starte Training ({'DRY-RUN' if args.dry_run else 'FULL'})...")
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    shutil.copy2(args.tokenizer, output_dir / "tokenizer.model")
    print(f"Training abgeschlossen! Modell & Tokenizer gespeichert in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainiert das FUTO-DE Llama-Modell.")
    parser.add_argument("--corpus", type=Path, default=config.CORPUS_PATH)
    parser.add_argument("--tokenizer", type=Path, default=config.FALLBACK_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--xbu-prob", type=float, default=0.5)
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
