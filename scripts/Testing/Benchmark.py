"""FUTO-DE Benchmark für GGUF (via libllama.so) und HuggingFace-PyTorch-Modelle.

Testet sowohl Next-Word-Prediction als auch FUTOs <XBU>-Autokorrektur-Protokoll
über 11 erweiterte Kategorien inkl. Case-Sensitivity-Auswertung und JSON-Export.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_bool,
    c_char_p,
    c_float,
    c_int8,
    c_int32,
    c_size_t,
    c_uint32,
    c_void_p,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import gguf
import numpy as np
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config

BLACKLIST_SYMBOLS = set(".!@#$%^&*()_=?/,\\][{};:\"><+`~|\r\n\t\x0b\x0c")


# --- ctypes-Definitionen passend zu llama.cpp/include/llama.h ---

class LlamaBatch(Structure):
    _fields_ = [
        ("n_tokens", c_int32),
        ("token", POINTER(c_int32)),
        ("embd", POINTER(c_float)),
        ("pos", POINTER(c_int32)),
        ("n_seq_id", POINTER(c_int32)),
        ("seq_id", POINTER(POINTER(c_int32))),
        ("logits", POINTER(c_int8)),
    ]


class LlamaModelParams(Structure):
    _fields_ = [
        ("devices", c_void_p),
        ("tensor_buft_overrides", c_void_p),
        ("n_gpu_layers", c_int32),
        ("split_mode", c_int32),
        ("load_mode", c_int32),
        ("main_gpu", c_int32),
        ("tensor_split", POINTER(c_float)),
        ("progress_callback", c_void_p),
        ("progress_callback_user_data", c_void_p),
        ("kv_overrides", c_void_p),
        ("vocab_only", c_bool),
        ("check_tensors", c_bool),
        ("use_extra_bufts", c_bool),
        ("no_host", c_bool),
        ("no_alloc", c_bool),
        ("load_mtp", c_bool),
    ]


class LlamaContextParams(Structure):
    _fields_ = [
        ("n_ctx", c_uint32),
        ("n_batch", c_uint32),
        ("n_ubatch", c_uint32),
        ("n_seq_max", c_uint32),
        ("n_rs_seq", c_uint32),
        ("n_outputs_max", c_uint32),
        ("n_outputs_max_per_seq", c_uint32),
        ("n_threads", c_int32),
        ("n_threads_batch", c_int32),
        ("ctx_type", c_int32),
        ("rope_scaling_type", c_int32),
        ("pooling_type", c_int32),
        ("attention_type", c_int32),
        ("flash_attn_type", c_int32),
        ("rope_freq_base", c_float),
        ("rope_freq_scale", c_float),
        ("yarn_ext_factor", c_float),
        ("yarn_attn_factor", c_float),
        ("yarn_beta_fast", c_float),
        ("yarn_beta_slow", c_float),
        ("yarn_orig_ctx", c_uint32),
        ("defrag_thold", c_float),
        ("cb_eval", c_void_p),
        ("cb_eval_user_data", c_void_p),
        ("type_k", c_int32),
        ("type_v", c_int32),
        ("abort_callback", c_void_p),
        ("abort_callback_data", c_void_p),
        ("embeddings", c_bool),
        ("offload_kqv", c_bool),
        ("no_perf", c_bool),
        ("op_offload", c_bool),
        ("swa_full", c_bool),
        ("kv_unified", c_bool),
        ("samplers", c_void_p),
        ("n_samplers", c_size_t),
        ("ctx_other", c_void_p),
    ]


_LOG_CB_TYPE = CFUNCTYPE(None, c_int32, c_char_p, c_void_p)
_NOOP_LOG_CB = _LOG_CB_TYPE(lambda _level, _text, _ud: None)


def _load_llama_lib() -> ctypes.CDLL:
    bin_dir = config.LLAMA_DIR / "build" / "bin"
    if not (bin_dir / "libllama.so").exists():
        raise FileNotFoundError(
            f"libllama.so nicht gefunden unter {bin_dir}. Bitte llama.cpp bauen."
        )
    for dep in ("libggml-base.so", "libggml-cpu.so", "libggml.so"):
        dep_path = bin_dir / dep
        if dep_path.exists():
            ctypes.CDLL(str(dep_path), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(bin_dir / "libllama.so"), mode=ctypes.RTLD_GLOBAL)

    lib.llama_log_set.argtypes = [_LOG_CB_TYPE, c_void_p]
    lib.llama_log_set.restype = None
    lib.llama_backend_init.argtypes = []
    lib.llama_backend_init.restype = None
    lib.llama_backend_free.argtypes = []
    lib.llama_backend_free.restype = None

    lib.llama_model_default_params.argtypes = []
    lib.llama_model_default_params.restype = LlamaModelParams
    lib.llama_model_load_from_file.argtypes = [c_char_p, LlamaModelParams]
    lib.llama_model_load_from_file.restype = c_void_p
    lib.llama_model_free.argtypes = [c_void_p]
    lib.llama_model_free.restype = None

    lib.llama_context_default_params.argtypes = []
    lib.llama_context_default_params.restype = LlamaContextParams
    lib.llama_init_from_model.argtypes = [c_void_p, LlamaContextParams]
    lib.llama_init_from_model.restype = c_void_p
    lib.llama_free.argtypes = [c_void_p]
    lib.llama_free.restype = None

    lib.llama_model_get_vocab.argtypes = [c_void_p]
    lib.llama_model_get_vocab.restype = c_void_p
    lib.llama_vocab_n_tokens.argtypes = [c_void_p]
    lib.llama_vocab_n_tokens.restype = c_int32

    lib.llama_get_memory.argtypes = [c_void_p]
    lib.llama_get_memory.restype = c_void_p
    lib.llama_memory_clear.argtypes = [c_void_p, c_bool]
    lib.llama_memory_clear.restype = None

    lib.llama_batch_get_one.argtypes = [POINTER(c_int32), c_int32]
    lib.llama_batch_get_one.restype = LlamaBatch
    lib.llama_decode.argtypes = [c_void_p, LlamaBatch]
    lib.llama_decode.restype = c_int32
    lib.llama_get_logits_ith.argtypes = [c_void_p, c_int32]
    lib.llama_get_logits_ith.restype = POINTER(c_float)

    lib.llama_log_set(_NOOP_LOG_CB, None)
    lib.llama_backend_init()
    return lib


class InferenceBackend(Protocol):
    sp: spm.SentencePieceProcessor
    vocab_size: int
    name: str

    def get_logits(self, token_ids: list[int]) -> np.ndarray: ...
    def close(self) -> None: ...


class LibLlamaBackend:
    """GGUF-Inferenz direkt über llama.cpp/build/bin/libllama.so."""

    def __init__(self, model_path: Path, n_ctx: int = 512, n_threads: int = 4):
        self.name = model_path.name
        self.lib = _load_llama_lib()
        self.sp = self._load_embedded_spm(model_path)

        mparams = self.lib.llama_model_default_params()
        mparams.n_gpu_layers = 0
        self.model = self.lib.llama_model_load_from_file(str(model_path).encode("utf-8"), mparams)
        if not self.model:
            raise RuntimeError(f"libllama konnte GGUF nicht laden: {model_path}")

        cparams = self.lib.llama_context_default_params()
        cparams.n_ctx = n_ctx
        cparams.n_batch = n_ctx
        cparams.n_ubatch = n_ctx
        cparams.n_threads = n_threads
        cparams.n_threads_batch = n_threads
        self.ctx = self.lib.llama_init_from_model(self.model, cparams)
        if not self.ctx:
            self.lib.llama_model_free(self.model)
            raise RuntimeError(f"llama_init_from_model fehlgeschlagen: {model_path}")

        vocab = self.lib.llama_model_get_vocab(self.model)
        self.vocab_size = int(self.lib.llama_vocab_n_tokens(vocab))
        self.mem = self.lib.llama_get_memory(self.ctx)

    @staticmethod
    def _load_embedded_spm(model_path: Path) -> spm.SentencePieceProcessor:
        reader = gguf.GGUFReader(str(model_path))
        field = reader.fields.get("keyboardlm.ext_tokenizer_data")
        if field is not None:
            spm_bytes = bytes(field.contents())
            return spm.SentencePieceProcessor(model_proto=spm_bytes)
        return spm.SentencePieceProcessor(model_file=str(config.FALLBACK_TOKENIZER_PATH))

    def get_logits(self, token_ids: list[int]) -> np.ndarray:
        self.lib.llama_memory_clear(self.mem, True)
        arr = (c_int32 * len(token_ids))(*token_ids)
        batch = self.lib.llama_batch_get_one(arr, len(token_ids))
        rc = self.lib.llama_decode(self.ctx, batch)
        if rc != 0:
            raise RuntimeError(f"llama_decode fehlgeschlagen (rc={rc})")
        ptr = self.lib.llama_get_logits_ith(self.ctx, -1)
        return np.ctypeslib.as_array(ptr, shape=(self.vocab_size,)).copy()

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.lib.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            self.lib.llama_model_free(self.model)
            self.model = None


class PyTorchBackend:
    """Reine PyTorch/Transformers-Inferenz ohne llama.cpp."""

    def __init__(self, model_dir: Path, device: str = "cpu"):
        import torch
        from transformers import LlamaForCausalLM

        self.torch = torch
        self.device = device
        self.name = f"{model_dir.name} (PyTorch/{device})"
        tok_path = model_dir / "tokenizer.model"
        if not tok_path.exists():
            tok_path = config.FALLBACK_TOKENIZER_PATH
        self.sp = spm.SentencePieceProcessor(model_file=str(tok_path))
        self.model = LlamaForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32)
        self.model.to(device)
        self.model.eval()
        self.vocab_size = int(self.model.config.vocab_size)

    def get_logits(self, token_ids: list[int]) -> np.ndarray:
        with self.torch.no_grad():
            inp = self.torch.tensor([token_ids], dtype=self.torch.long, device=self.device)
            out = self.model(input_ids=inp)
            return out.logits[0, -1, :].float().cpu().numpy()

    def close(self) -> None:
        self.model = None


def open_backend(path: Path, threads: int = 4, device: str = "cpu") -> InferenceBackend:
    if path.is_dir():
        return PyTorchBackend(path, device=device)
    return LibLlamaBackend(path, n_threads=threads)


class FutoPredictor:
    """FUTO-nahe Beam-Search-Dekodierung für Next-Word und <XBU>-Autokorrektur."""

    def __init__(self, backend: InferenceBackend):
        self.backend = backend
        self.sp = backend.sp
        self.vocab_size = backend.vocab_size
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.space_id = self.sp.piece_to_id("▁")
        self.dash_id = self.sp.piece_to_id("-")
        self.xbu_id = self.sp.piece_to_id("<XBU>")
        self.xbc_id = self.sp.piece_to_id("<XBC>")
        self.xec_id = self.sp.piece_to_id("<XEC>")
        self.char_a_id = self.sp.piece_to_id("<CHAR_A>")

        self.supports_xbu = (
            self.xbu_id != self.unk_id
            and self.xbc_id != self.unk_id
            and self.xec_id != self.unk_id
            and self.char_a_id != self.unk_id
            and all(
                self.sp.piece_to_id(f"<CHAR_{chr(ord('A') + i)}>") == self.char_a_id + i
                for i in range(26)
            )
        )

        banned = np.zeros(self.vocab_size, dtype=bool)
        for tid in range(min(self.sp.vocab_size(), self.vocab_size)):
            if self.sp.is_control(tid) or self.sp.is_unknown(tid) or self.sp.is_unused(tid):
                banned[tid] = True
                continue
            piece = self.sp.id_to_piece(tid)
            if any(c in BLACKLIST_SYMBOLS for c in piece):
                banned[tid] = True
            if piece in {"-▁", "<XBU>", "<XBC>", "<XEC>"} or piece.startswith("<CHAR_"):
                banned[tid] = True
        self.base_banned = banned

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        x = logits.astype(np.float64)
        x -= np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def _is_word_end(self, tid: int) -> bool:
        return self.sp.id_to_piece(tid).endswith("▁")

    def _beam_search(
        self, prompt_ids: list[int], top_k: int = 3, allow_xec: bool = False, max_tokens: int = 6
    ) -> list[tuple[str, float]]:
        probs = self._softmax(self.backend.get_logits(prompt_ids))
        probs[self.base_banned] = 0.0
        if self.space_id != self.unk_id:
            probs[self.space_id] = 0.0
        if self.dash_id != self.unk_id:
            probs[self.dash_id] = 0.0

        top_ids = np.argpartition(probs, -top_k)[-top_k:]
        top_ids = top_ids[np.argsort(probs[top_ids])[::-1]]

        active: list[tuple[float, list[int]]] = []
        outputs: list[tuple[float, list[int]]] = []
        for tid in top_ids:
            tid_int = int(tid)
            score = float(probs[tid_int])
            if score <= 0.0:
                continue
            if self._is_word_end(tid_int):
                outputs.append((score, [tid_int]))
            else:
                active.append((score, [tid_int]))

        for _ in range(1, max_tokens):
            remaining = top_k - len(outputs)
            if remaining <= 0 or not active:
                break
            next_candidates: list[tuple[float, list[int]]] = []
            for parent_score, seq in active:
                step_probs = self._softmax(self.backend.get_logits(prompt_ids + seq))
                xec_prob = float(step_probs[self.xec_id]) if allow_xec else 0.0
                step_probs[self.base_banned] = 0.0
                if allow_xec:
                    step_probs[self.xec_id] = xec_prob
                if seq[-1] == self.dash_id and self.dash_id != self.unk_id:
                    step_probs[self.dash_id] = 0.0

                cand_ids = np.argpartition(step_probs, -remaining)[-remaining:]
                for tid in cand_ids:
                    tid_int = int(tid)
                    p = float(step_probs[tid_int])
                    if p > 0.0:
                        next_candidates.append((parent_score * p, seq + [tid_int]))

            next_candidates.sort(key=lambda item: item[0], reverse=True)
            active = []
            for score, seq in next_candidates[:remaining]:
                last = seq[-1]
                if allow_xec and last == self.xec_id:
                    outputs.append((score, seq[:-1]))
                elif self._is_word_end(last):
                    outputs.append((score, seq))
                else:
                    active.append((score, seq))

        for score, seq in active:
            if len(outputs) < top_k:
                outputs.append((score, seq))

        outputs.sort(key=lambda item: item[0], reverse=True)
        dedup: list[tuple[str, float]] = []
        seen_words: set[str] = set()
        for score, seq in outputs:
            word = self.sp.decode_ids(seq).strip()
            if word and word not in seen_words:
                seen_words.add(word)
                dedup.append((word, score))
        return dedup[:top_k]

    def predict_next_word(self, context: str, top_k: int = 3) -> list[tuple[str, float]]:
        if not context.strip():
            prompt_ids = [self.bos_id]
        else:
            prompt_text = context.strip() if context.endswith("-") else f"{context.strip()} "
            prompt_ids = [self.bos_id] + self.sp.encode_as_ids(prompt_text)
        return self._beam_search(prompt_ids, top_k=top_k, allow_xec=False)

    def predict_xbu(self, context: str, typed: str, top_k: int = 3) -> list[tuple[str, float]]:
        if not self.supports_xbu:
            return []
        prompt_ids = [self.bos_id]
        if context.strip():
            prompt_ids.extend(self.sp.encode_as_ids(f"{context.strip()} "))
        prompt_ids.append(self.xbu_id)
        for ch in typed.lower():
            if "a" <= ch <= "z":
                prompt_ids.append(self.char_a_id + (ord(ch) - ord("a")))
        prompt_ids.append(self.xbc_id)
        return self._beam_search(prompt_ids, top_k=top_k, allow_xec=True)


# --- Erweiterte kuratierte Test-Suite (165 Fälle in 11 Kategorien) ---

TEST_SUITE: dict[str, list[tuple]] = {
    "1. NW: Entitäten & Geografie": [
        ("next", "Costa", ["Rica"]),
        ("next", "San", ["Francisco", "Diego"]),
        ("next", "New", ["York"]),
        ("next", "Deutsche", ["Bahn", "Telekom", "Post", "Bank", "Welle"]),
        ("next", "Bundesrepublik", ["Deutschland"]),
        ("next", "Buenos", ["Aires"]),
        ("next", "Vereinigte", ["Staaten", "Nationen"]),
        ("next", "Nordrhein-", ["Westfalen"]),
        ("next", "Baden-", ["Württemberg"]),
        ("next", "Los", ["Angeles"]),
        ("next", "Europäische", ["Union", "Kommission", "Zentralbank"]),
        ("next", "Brandenburger", ["Tor"]),
    ],
    "2. NW: Messenger & Chat": [
        ("next", "Guten", ["Morgen", "Tag", "Abend"]),
        ("next", "Ich melde mich", ["gleich", "später", "nachher", "morgen", "noch", "wieder"]),
        ("next", "Bin gerade", ["unterwegs", "noch", "im", "beim", "dabei", "auf"]),
        ("next", "Wann bist du", ["da", "zu", "fertig", "wieder", "zuhause", "morgen"]),
        ("next", "Schönes", ["Wochenende", "Fest"]),
        ("next", "Bis", ["gleich", "morgen", "später", "dann", "bald"]),
        ("next", "Kein", ["Problem", "Ding"]),
        ("next", "Hast du heute Abend", ["Zeit", "Lust", "schon", "was", "etwas"]),
        ("next", "Ich bin gleich", ["da", "fertig", "wieder", "zuhause", "bei"]),
        ("next", "Gib mir kurz", ["Bescheid", "Zeit", "noch", "ein"]),
        ("next", "Tut mir echt", ["leid", "Leid"]),
        ("next", "Sag mal, hast du", ["Lust", "Zeit", "schon", "eine", "ein", "eine"]),
        ("next", "Kommst du nachher noch", ["vorbei", "mit", "kurz", "rüber", "nach"]),
        ("next", "Wir können ja morgen", ["noch", "mal", "telefonieren", "sehen", "weiter", "wieder"]),
        ("next", "Gute", ["Nacht", "Besserung", "Idee", "Fahrt"]),
        ("next", "Lass uns später", ["nochmal", "noch", "kurz", "telefonieren", "schreiben", "reden"]),
        ("next", "Alles", ["klar", "gut", "Gute"]),
        ("next", "Danke", ["schön", "sehr", "dir", "euch", "für"]),
    ],
    "3. NW: Formell & E-Mail (Siezen)": [
        ("next", "Sehr geehrte", ["Damen", "Frau", "Herren"]),
        ("next", "Sehr geehrte Damen und", ["Herren"]),
        ("next", "Vielen Dank für Ihre", ["Nachricht", "Antwort", "Rückmeldung", "Hilfe", "Mühe", "E-Mail"]),
        ("next", "Mit freundlichen", ["Grüßen"]),
        ("next", "Bei Rückfragen stehe ich", ["Ihnen", "gerne"]),
        ("next", "Ich freue mich auf Ihre", ["Rückmeldung", "Antwort", "Nachricht"]),
        ("next", "Im Anhang finden", ["Sie"]),
        ("next", "Könnten Sie mir bitte", ["mitteilen", "sagen", "helfen", "ein", "eine", "noch"]),
        ("next", "Wir wünschen Ihnen ein schönes", ["Wochenende", "Fest", "Jahr"]),
        ("next", "Für Ihr Verständnis vielen", ["Dank"]),
        ("next", "Ich bitte um kurze", ["Rückmeldung", "Bestätigung", "Antwort", "Info"]),
        ("next", "Herzlichen Dank im", ["Voraus"]),
    ],
    "4. NW: Grammatik (Kasus/Genus/das-dass)": [
        ("next", "Ich bleibe heute daheim, weil ich krank", ["bin", "geworden", "war"]),
        ("next", "Er fährt morgen mit dem", ["Zug", "Auto", "Bus", "Fahrrad", "Rad"]),
        ("next", "Trotz des schlechten", ["Wetters"]),
        ("next", "Wir haben uns gestern lange", ["unterhalten", "gesehen", "getroffen"]),
        ("next", "Ich hoffe, dass sie morgen", ["kommt", "kommen", "wieder", "da"]),
        ("next", "Ich freue mich schon auf das", ["Wochenende", "Treffen", "Spiel", "Konzert", "Essen"]),
        ("next", "Kannst du mir bitte", ["helfen", "sagen", "mal", "ein", "eine"]),
        ("next", "Ich weiß nicht, ob ich das", ["schaffe", "will", "kann", "richtig", "mache", "so"]),
        ("next", "Es freut mich, dass du", ["da", "es", "so", "wieder", "gekommen", "dich"]),
        ("next", "Sie hat vergessen, mir zu", ["schreiben", "antworten", "sagen", "helfen", "danken"]),
        ("next", "Wegen der hohen", ["Kosten", "Preise", "Nachfrage", "Temperaturen"]),
        ("next", "Während des gesamten", ["Tages", "Jahres", "Wochenendes", "Urlaubs", "Zeitraums"]),
        ("next", "Er hat das Buch auf den", ["Tisch", "Boden", "Schrank", "Schreibtisch"]),
        ("next", "Sie sitzt schon seit einer", ["Stunde", "Woche", "halben", "Weile"]),
        ("next", "Das ist das Beste, was mir", ["passiert", "je", "heute", "bisher"]),
        ("next", "Seid ihr alle gut nach Hause", ["gekommen"]),
        ("next", "Ich habe mir vorgenommen, mehr", ["Zeit", "Sport", "zu", "Wasser"]),
        ("next", "Obwohl er müde war, hat", ["er"]),
    ],
    "5. NW: Alltag & Redewendungen": [
        ("next", "Vielen Dank für", ["die", "deine", "das", "Ihre"]),
        ("next", "Im Großen und", ["Ganzen"]),
        ("next", "Hin und", ["wieder", "her", "zurück"]),
        ("next", "Ab und", ["zu", "an"]),
        ("next", "Unter anderem", ["auch", "für", "bei", "mit", "deshalb"]),
        ("next", "Meiner Meinung", ["nach"]),
        ("next", "Auf jeden", ["Fall"]),
        ("next", "Am besten wäre", ["es"]),
        ("next", "Das macht keinen", ["Sinn", "Unterschied", "Spaß"]),
        ("next", "Ehrlich gesagt habe ich", ["keine", "noch", "da", "es", "mich"]),
        ("next", "Es kommt ganz darauf", ["an"]),
        ("next", "Ich habe heute keine", ["Zeit", "Lust", "Ahnung", "Kraft"]),
        ("next", "Von mir aus", ["gerne", "kann", "auch", "ja"]),
        ("next", "Ich halte dich auf dem", ["Laufenden"]),
        ("next", "Das ist mir völlig", ["egal", "klar", "neu", "gleich"]),
    ],
    "6. XBU: Präfix-Vervollständigung": [
        ("xbu", "Ich fahre zum", "hauptb", ["Hauptbahnhof"]),
        ("xbu", "Hast du meine", "nachr", ["Nachricht", "Nachrichten"]),
        ("xbu", "Wir müssen noch", "eink", ["einkaufen"]),
        ("xbu", "Können wir kurz", "telefon", ["telefonieren"]),
        ("xbu", "Ich wünsche dir ein schönes", "wochen", ["Wochenende"]),
        ("xbu", "Das klappt", "wahrsche", ["wahrscheinlich"]),
        ("xbu", "Herzlichen", "gluckw", ["Glückwunsch"]),
        ("xbu", "Das war eine große", "uberrasch", ["Überraschung"]),
        ("xbu", "Ich muss noch schnell eine", "entscheid", ["Entscheidung"]),
        ("xbu", "Bitte schick mir die", "unterl", ["Unterlagen"]),
        ("xbu", "Wir treffen uns am", "donner", ["Donnerstag"]),
        ("xbu", "Der Termin findet im", "septemb", ["September"]),
        ("xbu", "Ich habe leider keine", "inform", ["Informationen", "Information"]),
        ("xbu", "Das ist eine gute", "moglichk", ["Möglichkeit"]),
        ("xbu", "Kannst du die", "temperat", ["Temperatur"]),
        ("xbu", "Er arbeitet in der", "verwal", ["Verwaltung"]),
        ("xbu", "Wir brauchen noch", "unterst", ["Unterstützung"]),
        ("xbu", "Es gibt ein", "problem", ["Problem", "Probleme"]),
    ],
    "7. XBU: Präfix + Tippfehler": [
        ("xbu", "Ich fahre zum", "huaptb", ["Hauptbahnhof"]),
        ("xbu", "Hast du meine", "nahcr", ["Nachricht", "Nachrichten"]),
        ("xbu", "Das klappt", "wahrshc", ["wahrscheinlich"]),
        ("xbu", "Ich wünsche dir ein schönes", "wcohen", ["Wochenende"]),
        ("xbu", "Können wir kurz", "telrfon", ["telefonieren"]),
        ("xbu", "Bitte schick mir die", "utnerl", ["Unterlagen"]),
        ("xbu", "Wir treffen uns am", "dnnerst", ["Donnerstag"]),
        ("xbu", "Wir müssen noch", "ienkauf", ["einkaufen"]),
        ("xbu", "Das ist eine wichtige", "entschdi", ["Entscheidung"]),
        ("xbu", "Vielen Dank für die", "utnerst", ["Unterstützung"]),
        ("xbu", "Ich brauche noch mehr", "infromat", ["Informationen"]),
        ("xbu", "Das ist eine echte", "uberrash", ["Überraschung"]),
        ("xbu", "Der Zug hat heute", "verspaet", ["Verspätung"]),
        ("xbu", "Es gibt ein kleines", "porbl", ["Problem"]),
    ],
    "8. XBU: Tippfehler-Autokorrektur": [
        ("xbu", "Ich komme", "heuet", ["heute"]),
        ("xbu", "Das ist", "vieleicht", ["vielleicht"]),
        ("xbu", "Bis", "mrogn", ["morgen"]),
        ("xbu", "Schönes", "wcohenende", ["Wochenende"]),
        ("xbu", "Das stimmt", "nciht", ["nicht"]),
        ("xbu", "Vielen", "dnka", ["Dank"]),
        ("xbu", "Gute", "iede", ["Idee"]),
        ("xbu", "Ich habe keine", "ahnugn", ["Ahnung"]),
        ("xbu", "Das ist wirklich", "scher", ["schwer", "sehr"]),
        ("xbu", "Kannst du mir das bitte", "erklaern", ["erklären"]),
        ("xbu", "Ich freue mich", "riesgi", ["riesig"]),
        ("xbu", "Wir sehen uns", "spaetr", ["später"]),
        ("xbu", "Das Wetter ist heute", "wunderschoe", ["wunderschön"]),
        ("xbu", "Ich muss morgen früh", "aufstehn", ["aufstehen"]),
        ("xbu", "Das habe ich mir", "gedcaht", ["gedacht"]),
        ("xbu", "Entschuldige die späte", "antowrt", ["Antwort"]),
        ("xbu", "Alles klar, bis", "glecih", ["gleich"]),
        ("xbu", "Hast du das schon", "gesehn", ["gesehen"]),
    ],
    "9. XBU: Deutsche Umlaute & ß": [
        ("xbu", "Vielen Dank", "fur", ["für"]),
        ("xbu", "Wir", "konnen", ["können"]),
        ("xbu", "Ich freue mich", "uber", ["über"]),
        ("xbu", "Er wohnt in der", "strasse", ["Straße"]),
        ("xbu", "Ein", "schones", ["schönes"]),
        ("xbu", "Wir", "knnen", ["können"]),
        ("xbu", "Das ist", "fr", ["für"]),
        ("xbu", "Die richtige", "groe", ["Größe"]),
        ("xbu", "Ich hätte gerne einen Kaffee und ein", "stuck", ["Stück"]),
        ("xbu", "Das müssen wir noch", "klaren", ["klären"]),
        ("xbu", "Wir haben uns schon lange nicht mehr", "gehort", ["gehört"]),
        ("xbu", "Es wird langsam wieder", "kalter", ["kälter"]),
        ("xbu", "Draußen ist es heute ziemlich", "heiss", ["heiß"]),
        ("xbu", "Er fährt gerne mit dem", "zug", ["Zug"]),
        ("xbu", "Das ist die beste", "losung", ["Lösung"]),
        ("xbu", "Bis", "spater", ["später"]),
    ],
    "10. XBU: Satzanfang (ohne Kontext)": [
        ("xbu", "", "guten", ["Guten", "guten"]),
        ("xbu", "", "hallo", ["Hallo"]),
        ("xbu", "", "vielen", ["Vielen"]),
        ("xbu", "", "ich", ["Ich"]),
        ("xbu", "", "danke", ["Danke"]),
        ("xbu", "", "entschuldigung", ["Entschuldigung"]),
        ("xbu", "", "kannst", ["Kannst"]),
        ("xbu", "", "wahrsche", ["Wahrscheinlich"]),
        ("xbu", "", "vielicht", ["Vielleicht"]),
        ("xbu", "", "schones", ["Schönes"]),
        ("xbu", "", "heuet", ["Heute"]),
        ("xbu", "", "ubrigens", ["Übrigens"]),
    ],
    "11. XBU: Identität & Groß-/Kleinschreibung": [
        ("xbu", "Ich gehe", "heute", ["heute"]),
        ("xbu", "Wir bleiben zu", "hause", ["Hause"]),
        ("xbu", "Bis", "morgen", ["morgen"]),
        ("xbu", "Wir gehen", "spazieren", ["spazieren"]),
        ("xbu", "Ich habe", "hunger", ["Hunger"]),
        ("xbu", "Das macht keinen", "sinn", ["Sinn"]),
        ("xbu", "Am Ende des", "tages", ["Tages"]),
        ("xbu", "Er trinkt morgens gerne einen", "kaffee", ["Kaffee"]),
        ("xbu", "Wir müssen das", "essen", ["essen", "Essen"]),
        ("xbu", "Ich bin gleich wieder", "zuruck", ["zurück"]),
        ("xbu", "Das dauert nur eine", "minute", ["Minute"]),
        ("xbu", "Schönen", "abend", ["Abend"]),
    ],
}


@dataclass
class CategoryStats:
    name: str
    total: int = 0
    skipped: int = 0
    top1: int = 0
    top3: int = 0
    case_top1: int = 0
    case_top3: int = 0


@dataclass
class ModelSummary:
    label: str
    size_mb: float
    total: int
    top1: int
    top3: int
    case_top1: int
    case_top3: int
    avg_ms: float
    next_top1: int
    next_top3: int
    next_total: int
    xbu_top1: int
    xbu_top3: int
    xbu_total: int
    categories: list[dict]


def _is_match(pred: str, expected: list[str], case_sensitive: bool = False) -> bool:
    if case_sensitive:
        return any(pred == exp for exp in expected)
    pred_l = pred.lower()
    return any(pred_l == exp.lower() for exp in expected)


def benchmark_model(
    path: Path, threads: int, device: str, verbose: bool
) -> ModelSummary:
    backend = open_backend(path, threads=threads, device=device)
    predictor = FutoPredictor(backend)

    if path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
    else:
        size_mb = sum(f.stat().st_size for f in path.rglob("*.safetensors")) / (1024 * 1024)

    print(f"\n{'=' * 78}")
    print(f" MODELL: {backend.name} ({size_mb:.1f} MB) | XBU aktiv: {predictor.supports_xbu}")
    print(f"{'=' * 78}")

    cat_stats: list[CategoryStats] = []
    latencies: list[float] = []
    next_t1 = next_t3 = next_n = 0
    xbu_t1 = xbu_t3 = xbu_n = 0

    try:
        for category, cases in TEST_SUITE.items():
            st = CategoryStats(category)
            if verbose:
                print(f"\n** {category} **")
            for case in cases:
                mode = case[0]
                if mode == "xbu" and not predictor.supports_xbu:
                    st.skipped += 1
                    continue

                t0 = time.perf_counter()
                if mode == "next":
                    _, ctx, expected = case
                    preds = predictor.predict_next_word(ctx, top_k=3)
                    prompt_lbl = f"'{ctx} '"
                else:
                    _, ctx, typed, expected = case
                    preds = predictor.predict_xbu(ctx, typed, top_k=3)
                    prompt_lbl = f"'{ctx}' + <{typed}>"
                ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(ms)

                words = [w for w, _ in preds]
                hit1 = bool(words) and _is_match(words[0], expected, case_sensitive=False)
                hit3 = any(_is_match(w, expected, case_sensitive=False) for w in words)
                case1 = bool(words) and _is_match(words[0], expected, case_sensitive=True)
                case3 = any(_is_match(w, expected, case_sensitive=True) for w in words)

                st.total += 1
                st.top1 += int(hit1)
                st.top3 += int(hit3)
                st.case_top1 += int(case1)
                st.case_top3 += int(case3)
                if mode == "next":
                    next_n += 1
                    next_t1 += int(hit1)
                    next_t3 += int(hit3)
                else:
                    xbu_n += 1
                    xbu_t1 += int(hit1)
                    xbu_t3 += int(hit3)

                if verbose:
                    tag = "✓ T1" if hit1 else ("~ T3" if hit3 else "✗ FAIL")
                    print(
                        f"  {tag:<6} | {prompt_lbl:<42} -> {words} | "
                        f"Erwartet: {expected} ({ms:.1f} ms)"
                    )
            cat_stats.append(st)
    finally:
        backend.close()

    total = sum(s.total for s in cat_stats)
    top1 = sum(s.top1 for s in cat_stats)
    top3 = sum(s.top3 for s in cat_stats)
    case_top1 = sum(s.case_top1 for s in cat_stats)
    case_top3 = sum(s.case_top3 for s in cat_stats)
    avg_ms = sum(latencies) / len(latencies) if latencies else 0.0

    print("\n--- Kategorie-Auswertung ---")
    cat_dicts: list[dict] = []
    for s in cat_stats:
        if s.total == 0:
            print(f"  {s.name:<42} | übersprungen (kein <XBU>-Tokenizer)")
            continue
        p1 = s.top1 / s.total * 100
        p3 = s.top3 / s.total * 100
        c1 = s.case_top1 / s.total * 100
        print(
            f"  {s.name:<42} | Top-1: {s.top1:2}/{s.total:2} ({p1:5.1f}%) | "
            f"Top-3: {s.top3:2}/{s.total:2} ({p3:5.1f}%) | Case-T1: {c1:5.1f}%"
        )
        cat_dicts.append({
            "name": s.name,
            "total": s.total,
            "top1": s.top1,
            "top3": s.top3,
            "case_top1": s.case_top1,
            "case_top3": s.case_top3,
            "top1_pct": round(p1, 2),
            "top3_pct": round(p3, 2),
            "case_top1_pct": round(c1, 2),
        })

    print("-" * 78)
    if total:
        print(
            f"  GESAMT ({total} Tests) | Top-1: {top1}/{total} ({top1/total*100:.1f}%) | "
            f"Top-3: {top3}/{total} ({top3/total*100:.1f}%) | "
            f"Case-T1: {case_top1/total*100:.1f}% | Ø Latenz: {avg_ms:.1f} ms"
        )
    if next_n:
        print(
            f"  Next-Word ({next_n})    | Top-1: {next_t1}/{next_n} ({next_t1/next_n*100:.1f}%) | "
            f"Top-3: {next_t3}/{next_n} ({next_t3/next_n*100:.1f}%)"
        )
    if xbu_n:
        print(
            f"  XBU-Autokorrekt ({xbu_n}) | Top-1: {xbu_t1}/{xbu_n} ({xbu_t1/xbu_n*100:.1f}%) | "
            f"Top-3: {xbu_t3}/{xbu_n} ({xbu_t3/xbu_n*100:.1f}%)"
        )

    return ModelSummary(
        label=path.name,
        size_mb=round(size_mb, 2),
        total=total,
        top1=top1,
        top3=top3,
        case_top1=case_top1,
        case_top3=case_top3,
        avg_ms=round(avg_ms, 2),
        next_top1=next_t1,
        next_top3=next_t3,
        next_total=next_n,
        xbu_top1=xbu_t1,
        xbu_top3=xbu_t3,
        xbu_total=xbu_n,
        categories=cat_dicts,
    )


def resolve_default_model() -> Path:
    models_dir = config.OUTPUT_GGUF_PATH.parent
    candidates = [
        models_dir / "German-v1.5-q5km.gguf",
        models_dir / "German-v1-q5km.gguf",
        models_dir / "German-v1-q80.gguf",
        config.OUTPUT_GGUF_PATH,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    existing_ggufs = sorted(models_dir.glob("*.gguf"), key=lambda p: p.stat().st_size)
    if existing_ggufs:
        return existing_ggufs[0]
    return config.OUTPUT_GGUF_PATH


def resolve_targets(args: argparse.Namespace) -> list[Path]:
    if args.all:
        models_dir = config.OUTPUT_GGUF_PATH.parent
        targets = sorted(models_dir.glob("*.gguf"), key=lambda p: p.stat().st_size)
        if args.include_pt and config.MODEL_DIR.exists():
            targets.append(config.MODEL_DIR)
        return targets
    if args.models:
        return [Path(m) for m in args.models]
    return [resolve_default_model()]


def main() -> None:
    parser = argparse.ArgumentParser(description="FUTO-DE Benchmark (libllama.so & PyTorch).")
    parser.add_argument("models", nargs="*", help="Pfad(e) zu .gguf-Datei oder Modell-Ordner.")
    parser.add_argument("--all", action="store_true", help="Alle .gguf in Models/ testen.")
    parser.add_argument("--include-pt", action="store_true", help="Bei --all auch futo_de_final/ (PyTorch) testen.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="PyTorch-Device (cpu / cuda).")
    parser.add_argument("--summary-only", "-q", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None, help="Ergebnisse als JSON speichern.")
    args = parser.parse_args()

    targets = resolve_targets(args)
    if not targets:
        raise SystemExit("Keine Modelle zum Testen gefunden.")

    summaries = [
        benchmark_model(t, threads=args.threads, device=args.device, verbose=not args.summary_only)
        for t in targets
    ]

    if len(summaries) > 1:
        print(f"\n{'=' * 84}")
        print(" VERGLEICHSTABELLE")
        print(f"{'=' * 84}")
        print(
            f"  {'Modell':<28} | {'MB':>6} | {'Top-1':>7} | {'Top-3':>7} | "
            f"{'CaseT1':>6} | {'NW T1':>6} | {'XBU T1':>6} | {'ms':>6}"
        )
        print("-" * 84)
        for s in summaries:
            t1_p = f"{s.top1/s.total*100:5.1f}%" if s.total else "  n/a "
            t3_p = f"{s.top3/s.total*100:5.1f}%" if s.total else "  n/a "
            c1_p = f"{s.case_top1/s.total*100:5.1f}%" if s.total else "  n/a "
            nw_p = f"{s.next_top1/s.next_total*100:5.1f}%" if s.next_total else "  n/a "
            xbu_p = f"{s.xbu_top1/s.xbu_total*100:5.1f}%" if s.xbu_total else "  n/a "
            print(
                f"  {s.label:<28} | {s.size_mb:6.1f} | {t1_p:>7} | {t3_p:>7} | "
                f"{c1_p:>6} | {nw_p:>6} | {xbu_p:>6} | {s.avg_ms:6.1f}"
            )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "label": s.label,
                "size_mb": s.size_mb,
                "total": s.total,
                "top1": s.top1,
                "top3": s.top3,
                "top1_pct": round(s.top1 / s.total * 100, 2) if s.total else 0.0,
                "top3_pct": round(s.top3 / s.total * 100, 2) if s.total else 0.0,
                "case_top1_pct": round(s.case_top1 / s.total * 100, 2) if s.total else 0.0,
                "case_top3_pct": round(s.case_top3 / s.total * 100, 2) if s.total else 0.0,
                "avg_ms": s.avg_ms,
                "next_top1": s.next_top1,
                "next_top3": s.next_top3,
                "next_total": s.next_total,
                "xbu_top1": s.xbu_top1,
                "xbu_top3": s.xbu_top3,
                "xbu_total": s.xbu_total,
                "categories": s.categories,
            }
            for s in summaries
        ]
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[✓] Benchmark-JSON gespeichert: {args.json_out}")


if __name__ == "__main__":
    main()
