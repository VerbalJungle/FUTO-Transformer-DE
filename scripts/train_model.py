"""Trainiert das FUTO-DE Sprachmodell mit allen v1.5-Upgrades:
1. Dynamische <XBU>-Augmentierung pro Epoche (Multi-Word & Satzanfang-Boost)
2. 2D-QWERTZ Soft-Embedding-Mixing (char_embed_mixing_v1) via inputs_embeds
3. Erweiterter FutoAugmenter inkl. Prefix + Typo
4. 4D-Dokument-Attention-Mask, per-Doc position_ids, adam_beta2=0.95, cosine_with_min_lr & eval NW/XBU Loss
5. Optionale Chat/Alltag Cooldown-Phase
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
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
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

WORD_SPAN_RE = re.compile(r"\S+\s*")
ALPHA_WORD_RE = re.compile(r"^[A-Za-zÄÖÜäöüß]{2,}$")

QWERTZ_NEIGHBORS = {
    "a": "qwsy", "b": "vghn", "c": "xdfv", "d": "serfxc", "e": "wsdr",
    "f": "drtgcv", "g": "ftzhvb", "h": "gzujbn", "i": "ujko", "j": "huiknm",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxy", "t": "rfgz",
    "u": "zhji", "v": "cfgb", "w": "qase", "x": "ysdc", "y": "asx",
    "z": "tghu",
}
ASCII_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Physische 2D-QWERTZ-Tastenkoordinaten für Soft-Embedding-Mixing (Row-Offset 0.0 / 0.35 / 0.85)
QWERTZ_COORDS: dict[str, tuple[float, float]] = {
    **{c: (i * 1.0, 0.0) for i, c in enumerate("qwertzuiop")},
    **{c: (0.35 + i * 1.0, 1.0) for i, c in enumerate("asdfghjkl")},
    **{c: (0.85 + i * 1.0, 2.0) for i, c in enumerate("yxcvbnm")},
}


class FutoAugmenter:
    """Erzeugt FUTO-<XBU>-Autokorrektur-Sequenzen und feingranulare Loss-Typen (0/1/2)."""

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
        """Simuliert FUTOs ASCII-Tastenstrom inkl. deutscher Umlaut-Varianten."""
        r = rng.random()
        if r < 0.55:
            repl = {"ä": "a", "ö": "o", "ü": "u", "ß": "ss"}
        elif r < 0.75:
            repl = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "s"}
        else:
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
        """Simuliert 25% Clean, 25% Prefix, 15% Prefix+Typo, 35% Full-Word Typo."""
        letters = self.word_to_ascii_letters(word, rng)
        if not letters:
            return []

        mode = rng.random()
        if mode < 0.25:
            typed = letters
        elif mode < 0.50 and len(letters) >= 4:
            # Kürzere Präfixe (2..6) leicht bevorzugen, passend zur mobilen Vervollständigung
            hi = min(len(letters) - 1, max(3, min(7, len(letters) - 1)))
            if len(letters) > hi + 1 and rng.random() < 0.30:
                hi = len(letters) - 1
            typed = letters[: rng.randint(2, hi)]
        elif mode < 0.65 and len(letters) >= 5:
            pref_len = rng.randint(3, len(letters) - 1)
            chars = self._single_typo(list(letters[:pref_len]), rng)
            typed = "".join(chars) or letters[:pref_len]
        else:
            chars = self._single_typo(list(letters), rng)
            if len(chars) >= 5 and rng.random() < 0.25:
                chars = self._single_typo(chars, rng)
            typed = "".join(chars) or letters

        return [self.char_a_id + (ord(c) - ord("a")) for c in typed if "a" <= c <= "z"]

    def encode_line(
        self, text: str, rng: random.Random, xbu_prob: float
    ) -> tuple[list[int], list[int]]:
        """Gibt (token_ids, loss_types) zurück: 0=maskiert, 1=Next-Word/EOS, 2=XBU-Ziel/XEC."""
        if xbu_prob <= 0.0 or rng.random() >= xbu_prob:
            ids = [self.bos_id] + self.sp.encode_as_ids(text) + [self.eos_id]
            ltypes = [0] + [1] * (len(ids) - 1)
            return ids, ltypes

        spans = list(WORD_SPAN_RE.finditer(text))
        candidates = [
            idx
            for idx, m in enumerate(spans)
            if ALPHA_WORD_RE.match(m.group(0).rstrip(" \t"))
        ]
        if not candidates:
            ids = [self.bos_id] + self.sp.encode_as_ids(text) + [self.eos_id]
            ltypes = [0] + [1] * (len(ids) - 1)
            return ids, ltypes

        # Satzanfang gezielt häufiger (25%) augmentieren; bei langen Sätzen 1-2 Wörter wählen
        if rng.random() < 0.25:
            first_idx = candidates[0]
        else:
            first_idx = rng.choice(candidates)

        chosen_indices = [first_idx]
        if len(candidates) >= 10 and rng.random() < 0.40:
            far = [c for c in candidates if abs(c - first_idx) >= 2]
            if far:
                chosen_indices.append(rng.choice(far))
        chosen_indices.sort()

        ids: list[int] = [self.bos_id]
        ltypes: list[int] = [0]
        cursor = 0

        for chosen_idx in chosen_indices:
            span = spans[chosen_idx]
            if span.start() > cursor:
                seg = text[cursor : span.start()]
                p_ids = self.sp.encode_as_ids(seg)
                ids.extend(p_ids)
                ltypes.extend([1] * len(p_ids))

            target_span = span.group(0)
            core_word = target_span.rstrip(" \t")
            char_ids = self.word_to_keystrokes(core_word, rng)
            t_ids = self.sp.encode_as_ids(target_span)

            if char_ids and t_ids:
                ids.append(self.xbu_id)
                ltypes.append(0)
                ids.extend(char_ids)
                ltypes.extend([0] * len(char_ids))
                ids.append(self.xbc_id)
                ltypes.append(0)
                ids.extend(t_ids)
                ltypes.extend([2] * len(t_ids))
                ids.append(self.xec_id)
                ltypes.append(2)
            else:
                ids.extend(t_ids)
                ltypes.extend([1] * len(t_ids))

            cursor = span.end()

        if cursor < len(text):
            s_ids = self.sp.encode_as_ids(text[cursor:])
            ids.extend(s_ids)
            ltypes.extend([1] * len(s_ids))

        ids.append(self.eos_id)
        ltypes.append(1)
        return ids, ltypes


def index_corpus_offsets(
    corpus_path: Path, max_lines: int = 0
) -> np.ndarray:
    """Scannt einmalig Byte-Offsets aller nicht-leeren Zeilen in ein uint64-Array."""
    offsets: list[int] = []
    with corpus_path.open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
                if max_lines > 0 and len(offsets) >= max_lines:
                    break
    return np.asarray(offsets, dtype=np.uint64)


class DynamicPackedCorpusDataset(Dataset):
    """Packt Sätze on-the-fly mit epochenabhängigem <XBU>-Augmentierungs-Seed."""

    def __init__(
        self,
        corpus_path: Path,
        offsets: np.ndarray,
        sp_model_path: Path,
        block_size: int,
        xbu_prob: float,
        is_eval: bool = False,
        base_seed: int = 42,
        max_blocks: int = 0,
        secondary_path: Path | None = None,
        secondary_offsets: np.ndarray | None = None,
        secondary_prob: float = 0.0,
    ):
        if len(offsets) == 0:
            raise RuntimeError(f"Keine Zeilen für Dataset aus {corpus_path}")
        self.corpus_path = corpus_path
        self.offsets = offsets
        self.sp_model_path = sp_model_path
        self.block_size = block_size
        self.xbu_prob = xbu_prob
        self.is_eval = is_eval
        self.base_seed = base_seed
        self.secondary_path = secondary_path
        self.secondary_offsets = secondary_offsets
        self.secondary_prob = secondary_prob

        self.shared_epoch = multiprocessing.Value("i", 0)
        self._worker_pid: int | None = None
        self._file = None
        self._sec_file = None
        self._augmenter: FutoAugmenter | None = None

        # Durchschnittliche Tokenlänge per kleiner Stichprobe schätzen
        aug = FutoAugmenter(spm.SentencePieceProcessor(model_file=str(sp_model_path)))
        sample_rng = random.Random(base_seed)
        sample_n = min(1_500, len(offsets))
        step = max(1, len(offsets) // sample_n)
        tok_total = 0
        sampled = 0
        with corpus_path.open("rb") as f:
            for i in range(0, len(offsets), step):
                f.seek(int(offsets[i]))
                txt = f.readline().decode("utf-8", errors="ignore").strip()
                if not txt:
                    continue
                ids, _ = aug.encode_line(txt, sample_rng, xbu_prob=xbu_prob)
                tok_total += len(ids)
                sampled += 1
                if sampled >= sample_n:
                    break
        avg_toks = (tok_total / sampled) if sampled else 24.0
        est_blocks = max(2, int((len(offsets) * avg_toks) / block_size))
        if max_blocks > 0:
            est_blocks = min(est_blocks, max_blocks)
        self.n_blocks = est_blocks

    def set_epoch(self, epoch: int) -> None:
        self.shared_epoch.value = int(epoch)

    def _ensure_worker_state(self) -> None:
        pid = os.getpid()
        if self._worker_pid != pid:
            if self._file is not None:
                self._file.close()
            if self._sec_file is not None:
                self._sec_file.close()
            self._file = self.corpus_path.open("rb")
            self._sec_file = (
                self.secondary_path.open("rb") if self.secondary_path is not None else None
            )
            sp = spm.SentencePieceProcessor(model_file=str(self.sp_model_path))
            self._augmenter = FutoAugmenter(sp)
            self._worker_pid = pid

    def _read_line(self, use_secondary: bool, row_idx: int) -> str:
        if use_secondary and self._sec_file is not None and self.secondary_offsets is not None:
            self._sec_file.seek(int(self.secondary_offsets[row_idx]))
            return self._sec_file.readline().decode("utf-8", errors="ignore").strip()
        assert self._file is not None
        self._file.seek(int(self.offsets[row_idx]))
        return self._file.readline().decode("utf-8", errors="ignore").strip()

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._ensure_worker_state()
        assert self._augmenter is not None

        epoch = 0 if self.is_eval else self.shared_epoch.value
        rng = random.Random(f"{self.base_seed}:{epoch}:{idx}")

        tok_buf: list[int] = []
        ltype_buf: list[int] = []
        doc_buf: list[int] = []
        doc_id = 0

        # Deterministischer Start-Cursor für Eval, damit Val-Sätze breit abgedeckt bleiben
        eval_cursor = (idx * max(1, len(self.offsets) // max(1, self.n_blocks))) % len(self.offsets)

        while len(tok_buf) < self.block_size:
            use_sec = (
                not self.is_eval
                and self._sec_file is not None
                and self.secondary_offsets is not None
                and len(self.secondary_offsets) > 0
                and rng.random() < self.secondary_prob
            )
            if use_sec:
                assert self.secondary_offsets is not None
                r_idx = rng.randrange(len(self.secondary_offsets))
                text = self._read_line(True, r_idx)
            elif self.is_eval:
                text = self._read_line(False, eval_cursor)
                eval_cursor = (eval_cursor + 1) % len(self.offsets)
            else:
                r_idx = rng.randrange(len(self.offsets))
                text = self._read_line(False, r_idx)

            if not text:
                continue
            ids, ltypes = self._augmenter.encode_line(text, rng, xbu_prob=self.xbu_prob)
            if not ids:
                continue
            tok_buf.extend(ids)
            ltype_buf.extend(ltypes)
            doc_buf.extend([doc_id] * len(ids))
            doc_id += 1

        return {
            "input_ids": torch.tensor(tok_buf[: self.block_size], dtype=torch.long),
            "loss_types": torch.tensor(ltype_buf[: self.block_size], dtype=torch.long),
            "doc_ids": torch.tensor(doc_buf[: self.block_size], dtype=torch.long),
        }


class FutoDataCollator:
    """Erstellt Labels, 4D-Dokument-Attention-Mask und per-Dokument position_ids."""

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([f["input_ids"] for f in features], dim=0)
        loss_types = torch.stack([f["loss_types"] for f in features], dim=0)
        doc_ids = torch.stack([f["doc_ids"] for f in features], dim=0)

        bsz, seq_len = input_ids.shape
        labels = input_ids.clone()
        labels[loss_types == 0] = -100

        # 4D Bool-Maske: True = darf attenden (gleiches Dokument + kausal)
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
        same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
        attention_mask = (same_doc & causal.unsqueeze(0)).unsqueeze(1)

        # Dokumentweise Positionen ab 0 zählen
        arange = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(bsz, seq_len)
        is_new_doc = torch.ones((bsz, seq_len), dtype=torch.bool)
        if seq_len > 1:
            is_new_doc[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
        doc_starts = torch.where(is_new_doc, arange, torch.zeros_like(arange)).cummax(dim=1).values
        position_ids = arange - doc_starts

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_types": loss_types,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }


class EpochUpdateCallback(TrainerCallback):
    """Aktualisiert den Epochen-Seed im dynamischen Dataset zu Beginn jeder Epoche."""

    def __init__(self, train_dataset: DynamicPackedCorpusDataset):
        self.train_dataset = train_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.train_dataset.set_epoch(int(state.epoch or 0))


class DivergenceGuardCallback(TrainerCallback):
    """Bricht frühzeitig ab, wenn Loss oder Grad-Norm auf NaN/Inf/0.0 divergieren."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss")
        if loss is not None and (not math.isfinite(loss) or (state.global_step >= 10 and loss == 0.0)):
            raise RuntimeError(f"Training divergiert bei Step {state.global_step}: logs={logs}")
        grad_norm = logs.get("grad_norm")
        if grad_norm is not None and not math.isfinite(grad_norm):
            raise RuntimeError(f"Grad-Norm nicht endlich bei Step {state.global_step}: logs={logs}")


class FutoTrainer(Trainer):
    """Trainer mit 2D-QWERTZ Soft-Embedding-Mixing und getrennter Eval-NW/XBU-Auswertung."""

    def __init__(
        self,
        *args,
        char_a_id: int,
        char_mix_prob: float = 0.65,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.char_a_id = char_a_id
        self.char_mix_prob = char_mix_prob
        coords = [QWERTZ_COORDS[chr(ord("a") + i)] for i in range(26)]
        self.key_coords = torch.tensor(coords, dtype=torch.float32)
        self._eval_stats = self._zero_eval_stats()

    @staticmethod
    def _zero_eval_stats() -> dict[str, float]:
        return {"nw_sum": 0.0, "nw_cnt": 0.0, "xbu_sum": 0.0, "xbu_cnt": 0.0}

    def _build_inputs_embeds(
        self, model: LlamaForCausalLM, input_ids: torch.Tensor
    ) -> torch.Tensor:
        emb_layer = model.get_input_embeddings()
        inputs_embeds = emb_layer(input_ids)

        if not model.training or self.char_mix_prob <= 0.0:
            return inputs_embeds

        is_char = (input_ids >= self.char_a_id) & (input_ids < self.char_a_id + 26)
        if not bool(is_char.any()):
            return inputs_embeds

        do_mix = is_char & (
            torch.rand(input_ids.shape, device=input_ids.device, dtype=torch.float32)
            < self.char_mix_prob
        )
        if not bool(do_mix.any()):
            return inputs_embeds

        coords = self.key_coords.to(device=input_ids.device)
        char_idx = input_ids[do_mix] - self.char_a_id
        base_xy = coords[char_idx]

        sigma = torch.empty((base_xy.shape[0], 1), device=input_ids.device).uniform_(0.22, 0.52)
        touch_xy = base_xy + torch.randn_like(base_xy) * sigma
        sq_dist = ((touch_xy[:, None, :] - coords[None, :, :]) ** 2).sum(dim=-1)

        neg_d, top_idx = torch.topk(-sq_dist, k=3, dim=-1)
        tau = 0.45
        weights = torch.softmax(neg_d / (2.0 * tau * tau), dim=-1)

        char_table = emb_layer.weight[self.char_a_id : self.char_a_id + 26]
        mixed = (weights.unsqueeze(-1) * char_table[top_idx]).sum(dim=1).to(inputs_embeds.dtype)

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[do_mix] = mixed
        return inputs_embeds

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_types = inputs.pop("loss_types", None)
        input_ids = inputs.pop("input_ids")
        inputs_embeds = self._build_inputs_embeds(model, input_ids)

        if not model.training and loss_types is not None and "labels" in inputs:
            labels = inputs.pop("labels")
            outputs = model(inputs_embeds=inputs_embeds, **inputs)
            with torch.no_grad():
                shift_logits = outputs.logits[:, :-1, :].contiguous().float()
                shift_labels = labels[:, 1:].contiguous()
                shift_types = loss_types[:, 1:].contiguous()
                per_tok = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(shift_labels)

                valid_mask = shift_labels != -100
                loss = (
                    per_tok[valid_mask].mean()
                    if bool(valid_mask.any())
                    else outputs.logits.sum() * 0.0
                )
                outputs.loss = loss

                nw_mask = (shift_types == 1) & valid_mask
                if bool(nw_mask.any()):
                    self._eval_stats["nw_sum"] += float(per_tok[nw_mask].sum().item())
                    self._eval_stats["nw_cnt"] += float(nw_mask.sum().item())

                xbu_mask = (shift_types == 2) & valid_mask
                if bool(xbu_mask.any()):
                    self._eval_stats["xbu_sum"] += float(per_tok[xbu_mask].sum().item())
                    self._eval_stats["xbu_cnt"] += float(xbu_mask.sum().item())

            return (loss, outputs) if return_outputs else loss

        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            inputs["num_items_in_batch"] = num_items_in_batch
        outputs = model(inputs_embeds=inputs_embeds, **inputs)
        loss = outputs.loss
        if (
            self.args.average_tokens_across_devices
            and self.model_accepts_loss_kwargs
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        self._eval_stats = self._zero_eval_stats()
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        nw_cnt = self._eval_stats["nw_cnt"]
        xbu_cnt = self._eval_stats["xbu_cnt"]
        extra: dict[str, float] = {}
        if nw_cnt > 0:
            extra[f"{metric_key_prefix}_nw_loss"] = round(self._eval_stats["nw_sum"] / nw_cnt, 4)
        if xbu_cnt > 0:
            extra[f"{metric_key_prefix}_xbu_loss"] = round(self._eval_stats["xbu_sum"] / xbu_cnt, 4)
        if extra:
            metrics.update(extra)
            print(f"  [Eval-Detail] {extra}")
        return metrics


def _split_train_val_offsets(
    offsets: np.ndarray, max_val_lines: int = 30_000, dry_run: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if len(offsets) < 2:
        raise RuntimeError("Korpus hat weniger als 2 Zeilen.")
    val_mask = (np.arange(len(offsets)) % 100) == 0
    val_offsets = offsets[val_mask]
    if len(val_offsets) > max_val_lines:
        val_offsets = val_offsets[:max_val_lines]
    train_offsets = offsets[~val_mask]
    if len(val_offsets) == 0 or len(train_offsets) == 0:
        val_offsets = offsets[:1]
        train_offsets = offsets[1:]
    if dry_run:
        train_offsets = train_offsets[: min(len(train_offsets), 400)]
        val_offsets = val_offsets[: min(len(val_offsets), 50)]
    return train_offsets, val_offsets


def _save_final_artifacts(trainer_obj: Trainer, output_dir: Path, tokenizer_path: Path, label: str) -> None:
    """Speichert Modell und Tokenizer sofort sauber ins Zielverzeichnis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_obj.save_model(str(output_dir))
    dst_tok = output_dir / "tokenizer.model"
    if tokenizer_path.resolve() != dst_tok.resolve():
        shutil.copy2(tokenizer_path, dst_tok)
    print(f"[✓] {label} gespeichert in: {output_dir}")


def _read_checkpoint_steps(ckpt_dir: Path, fallback: int = 13_000) -> int:
    state_path = ckpt_dir / "trainer_state.json"
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            step = int(data.get("global_step") or 0)
            if step > 0:
                return step
        except Exception:
            pass
    return fallback


def train(args: argparse.Namespace) -> None:
    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if not args.dry_run and not has_cuda:
        raise RuntimeError("FEHLER: ROCm/CUDA erkennt keine GPU! Training wird nicht auf CPU gestartet.")
    if has_cuda:
        torch.set_float32_matmul_precision("high")
        # Workaround für PyTorch 2.6 / ROCm 6.1 mem_efficient_sdp Backward-Bug bei 4D-Block-Masken
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU (dry-run)"
    print(f"Aktives Gerät: {device_name}")

    if not args.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer nicht gefunden: {args.tokenizer}")
    if not args.corpus.exists():
        raise FileNotFoundError(f"Korpus nicht gefunden: {args.corpus}")

    if args.cooldown_from is not None:
        if not args.cooldown_from.exists():
            raise FileNotFoundError(f"Phase-1-Checkpoint nicht gefunden: {args.cooldown_from}")
        if args.cooldown_corpus is None or not args.cooldown_corpus.exists():
            raise FileNotFoundError(f"Cooldown-Korpus nicht gefunden: {args.cooldown_corpus}")

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if not args.dry_run and sp.vocab_size() != config.VOCAB_SIZE:
        raise ValueError(
            f"Tokenizer vocab_size={sp.vocab_size()} stimmt nicht mit config.VOCAB_SIZE={config.VOCAB_SIZE} überein"
        )
    aug_check = FutoAugmenter(sp)

    output_dir = args.output_dir or (
        Path("/tmp/futo_dry_run_model") if args.dry_run else config.MODEL_DIR
    )
    ckpt_dir = output_dir.parent / f"{output_dir.name}_checkpoints"

    max_lines = args.max_lines if args.max_lines > 0 else (500 if args.dry_run else 0)
    all_offsets = index_corpus_offsets(args.corpus, max_lines=max_lines)
    train_offsets, val_offsets = _split_train_val_offsets(all_offsets, dry_run=args.dry_run)

    val_cap = 2 if args.dry_run else 1_000
    train_cap = 8 if args.dry_run else 0

    val_ds = DynamicPackedCorpusDataset(
        corpus_path=args.corpus,
        offsets=val_offsets,
        sp_model_path=args.tokenizer,
        block_size=args.block_size,
        xbu_prob=args.xbu_prob,
        is_eval=True,
        base_seed=99,
        max_blocks=val_cap,
    )

    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not use_bf16
    optim_name = "adamw_torch_fused" if has_cuda else "adamw_torch"
    num_workers = 0 if args.dry_run else 4
    persistent_workers = num_workers > 0
    prefetch_factor = 4 if num_workers > 0 else None
    collator = FutoDataCollator()

    if args.cooldown_from is not None:
        print(f"Lade Phase-1-Checkpoint für Cooldown: {args.cooldown_from}")
        model = LlamaForCausalLM.from_pretrained(str(args.cooldown_from))
        main_steps = _read_checkpoint_steps(args.cooldown_from, fallback=13_000)
        run_cooldown = True
    else:
        train_ds = DynamicPackedCorpusDataset(
            corpus_path=args.corpus,
            offsets=train_offsets,
            sp_model_path=args.tokenizer,
            block_size=args.block_size,
            xbu_prob=args.xbu_prob,
            is_eval=False,
            base_seed=42,
            max_blocks=train_cap,
        )
        print(
            f"[✓] Dynamische Datasets: Train={len(train_offsets):,} Zeilen (~{len(train_ds):,} Blöcke/Epoche), "
            f"Val={len(val_offsets):,} Zeilen ({len(val_ds):,} Blöcke)"
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

        training_args = TrainingArguments(
            output_dir=str(ckpt_dir),
            num_train_epochs=1 if args.dry_run else args.epochs,
            max_steps=2 if args.dry_run else -1,
            per_device_train_batch_size=2 if args.dry_run else args.batch_size,
            per_device_eval_batch_size=2 if args.dry_run else args.batch_size,
            gradient_accumulation_steps=1 if args.dry_run else args.grad_accum,
            learning_rate=args.lr,
            optim=optim_name,
            adam_beta1=0.9,
            adam_beta2=0.95,
            weight_decay=0.05,
            lr_scheduler_type="cosine_with_min_lr",
            lr_scheduler_kwargs={"min_lr_rate": 0.1},
            warmup_steps=0 if args.dry_run else 0.05,
            max_grad_norm=1.0,
            logging_steps=1 if args.dry_run else 100,
            eval_strategy="steps",
            eval_steps=1 if args.dry_run else 1500,
            save_strategy="steps",
            save_steps=2 if args.dry_run else 1500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            bf16=use_bf16,
            bf16_full_eval=use_bf16,
            fp16=use_fp16,
            use_cpu=not has_cuda,
            dataloader_num_workers=num_workers,
            dataloader_persistent_workers=persistent_workers,
            dataloader_prefetch_factor=prefetch_factor,
            dataloader_pin_memory=has_cuda,
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = FutoTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            callbacks=[EpochUpdateCallback(train_ds), DivergenceGuardCallback()],
            char_a_id=aug_check.char_a_id,
            char_mix_prob=args.char_mix_prob,
        )

        print(f"Starte Phase 1 Training ({'DRY-RUN' if args.dry_run else 'FULL'})...")
        trainer.train()
        main_steps = max(1, int(trainer.state.max_steps or trainer.state.global_step or 1))
        _save_final_artifacts(trainer, output_dir, args.tokenizer, label="Phase-1-Modell")
        del trainer
        if has_cuda:
            torch.cuda.empty_cache()

        run_cooldown = (
            not args.no_cooldown
            and args.cooldown_ratio > 0.0
            and args.cooldown_corpus is not None
            and args.cooldown_corpus.exists()
        )

    # Optionale Phase 2: Cooldown auf Chat-/Alltagssätzen
    if run_cooldown:
        cd_offsets = index_corpus_offsets(args.cooldown_corpus, max_lines=500 if args.dry_run else 0)
        if len(cd_offsets) >= 2:
            if args.dry_run:
                cd_steps = 2
            elif args.cooldown_steps > 0:
                cd_steps = args.cooldown_steps
            else:
                cd_steps = max(50, int(main_steps * args.cooldown_ratio))
            cd_ds = DynamicPackedCorpusDataset(
                corpus_path=args.corpus,
                offsets=train_offsets,
                sp_model_path=args.tokenizer,
                block_size=args.block_size,
                xbu_prob=args.xbu_prob,
                is_eval=False,
                base_seed=777,
                max_blocks=8 if args.dry_run else 0,
                secondary_path=args.cooldown_corpus,
                secondary_offsets=cd_offsets,
                secondary_prob=0.65,
            )
            cd_eval_steps = 1 if args.dry_run else max(50, cd_steps // 2)
            cd_save_steps = 2 if args.dry_run else max(250, cd_steps // 2)
            cd_args = TrainingArguments(
                output_dir=str(ckpt_dir.parent / f"{ckpt_dir.name}_cooldown"),
                max_steps=cd_steps,
                per_device_train_batch_size=2 if args.dry_run else args.batch_size,
                per_device_eval_batch_size=2 if args.dry_run else args.batch_size,
                gradient_accumulation_steps=1 if args.dry_run else args.grad_accum,
                learning_rate=args.lr * 0.15,
                optim=optim_name,
                adam_beta1=0.9,
                adam_beta2=0.95,
                weight_decay=0.05,
                lr_scheduler_type="cosine_with_min_lr",
                lr_scheduler_kwargs={"min_lr_rate": 0.15},
                warmup_steps=0,
                max_grad_norm=1.0,
                logging_steps=1 if args.dry_run else 50,
                eval_strategy="steps",
                eval_steps=cd_eval_steps,
                save_strategy="steps",
                save_steps=cd_save_steps,
                save_total_limit=2,
                bf16=use_bf16,
                bf16_full_eval=use_bf16,
                fp16=use_fp16,
                use_cpu=not has_cuda,
                dataloader_num_workers=num_workers,
                dataloader_persistent_workers=persistent_workers,
                dataloader_prefetch_factor=prefetch_factor,
                dataloader_pin_memory=has_cuda,
                remove_unused_columns=False,
                report_to="none",
            )
            cd_trainer = FutoTrainer(
                model=model,
                args=cd_args,
                train_dataset=cd_ds,
                eval_dataset=val_ds,
                data_collator=collator,
                callbacks=[EpochUpdateCallback(cd_ds), DivergenceGuardCallback()],
                char_a_id=aug_check.char_a_id,
                char_mix_prob=args.char_mix_prob,
            )
            print(
                f"Starte Phase 2 Cooldown ({cd_steps} Steps, LR={args.lr * 0.15:.2e}, "
                f"65% {args.cooldown_corpus.name})..."
            )
            cd_trainer.train()
            _save_final_artifacts(cd_trainer, output_dir, args.tokenizer, label="Finales Cooldown-Modell")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainiert das FUTO-DE v1.5 Llama-Modell.")
    parser.add_argument("--corpus", type=Path, default=config.CORPUS_PATH)
    parser.add_argument("--tokenizer", type=Path, default=config.FALLBACK_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None, help="(Kompatibilitätsflag, ungenutzt bei dynamischem Packing)")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--xbu-prob", type=float, default=0.60)
    parser.add_argument("--char-mix-prob", type=float, default=0.65)
    parser.add_argument("--cooldown-corpus", type=Path, default=config.COOLDOWN_CORPUS_PATH)
    parser.add_argument("--cooldown-ratio", type=float, default=0.10)
    parser.add_argument("--cooldown-from", type=Path, default=None, help="Phase-1-Checkpoint laden und direkt Phase 2 Cooldown trainieren.")
    parser.add_argument("--cooldown-steps", type=int, default=0, help="Explizite Step-Anzahl für Phase 2 (0 = automatisch aus Ratio).")
    parser.add_argument("--no-cooldown", action="store_true")
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
