"""Automatischer Held-out-Benchmark: Next-Word, Keystroke Savings Rate (KSR) & XBU-Stress."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from Benchmark import FutoPredictor, open_backend, resolve_default_model
from train_model import ASCII_LETTERS, QWERTZ_NEIGHBORS

WORD_RE = re.compile(r"\S+")
ALPHA_WORD_RE = re.compile(r"^[A-Za-zÄÖÜäöüß]{2,}$")


def load_heldout_records(
    path: Path, max_sentences: int, seed: int = 2026
) -> list[dict[str, str]]:
    """Lädt Held-out-Sätze stratifiziert (gleichmäßig über alle vorhandenen Quellen)."""
    if not path.exists():
        raise FileNotFoundError(f"Held-out-Datei nicht gefunden: {path}")
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                text = str(rec.get("text", "")).strip()
                source = str(rec.get("source", "unknown"))
            except json.JSONDecodeError:
                text = raw
                source = "text"
            if text:
                by_source[source].append({"source": source, "text": text})

    if not by_source:
        return []

    rng = random.Random(seed)
    sources = sorted(by_source.keys())
    for src in sources:
        rng.shuffle(by_source[src])

    total_available = sum(len(v) for v in by_source.values())
    limit = total_available if max_sentences <= 0 else min(max_sentences, total_available)

    # Round-Robin über alle Quellen, damit auch [:ksr_sentences] alle Domänen abdeckt
    records: list[dict[str, str]] = []
    idx_by_src = {src: 0 for src in sources}
    while len(records) < limit:
        added_any = False
        for src in sources:
            i = idx_by_src[src]
            if i < len(by_source[src]):
                records.append(by_source[src][i])
                idx_by_src[src] = i + 1
                added_any = True
                if len(records) >= limit:
                    break
        if not added_any:
            break
    return records


def _to_ascii_de(word: str) -> str:
    repl = {"ä": "a", "ö": "o", "ü": "u", "ß": "ss"}
    out = []
    for ch in word.lower():
        for m in repl.get(ch, ch):
            if "a" <= m <= "z":
                out.append(m)
    return "".join(out)


def _apply_single_typo(chars: list[str], rng: random.Random) -> list[str]:
    op = rng.random()
    if len(chars) >= 2 and op < 0.50:
        idx = rng.randrange(len(chars))
        nb = QWERTZ_NEIGHBORS.get(chars[idx], "")
        if nb:
            chars[idx] = rng.choice(nb)
    elif len(chars) >= 2 and op < 0.75:
        idx = rng.randrange(len(chars) - 1)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif len(chars) >= 3:
        chars.pop(rng.randrange(1, len(chars)))
    else:
        idx = rng.randrange(len(chars))
        chars.insert(idx, rng.choice(QWERTZ_NEIGHBORS.get(chars[idx], ASCII_LETTERS)))
    return chars


def _corrupt_word(word: str, mode: str, rng: random.Random) -> str | None:
    if mode == "umlaut":
        if not any(c in word.lower() for c in "äöüß"):
            return None
        if rng.random() < 0.6:
            repl = {"ä": "a", "ö": "o", "ü": "u", "ß": "ss"}
        else:
            repl = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "s"}
        out = "".join(repl.get(c, c) for c in word.lower())
        return "".join(c for c in out if "a" <= c <= "z") or None

    letters = _to_ascii_de(word)
    if len(letters) < 3:
        return None
    if mode == "typo_1":
        return "".join(_apply_single_typo(list(letters), rng))
    if mode == "typo_2" and len(letters) >= 5:
        c1 = _apply_single_typo(list(letters), rng)
        c2 = _apply_single_typo(c1, rng)
        return "".join(c2)
    if mode == "prefix_typo" and len(letters) >= 5:
        pref_len = rng.randint(3, max(3, len(letters) - 2))
        return "".join(_apply_single_typo(list(letters[:pref_len]), rng))
    return None


def eval_next_word(
    predictor: FutoPredictor, records: list[dict[str, str]], rng: random.Random
) -> dict:
    overall = {"n": 0, "t1": 0, "t3": 0, "mrr": 0.0}
    by_source: dict[str, dict] = defaultdict(lambda: {"n": 0, "t1": 0, "t3": 0, "mrr": 0.0})

    for rec in records:
        matches = list(WORD_RE.finditer(rec["text"]))
        valid_pos = [
            i
            for i, m in enumerate(matches)
            if i >= 1 and ALPHA_WORD_RE.match(m.group(0).rstrip(".,!?;:\"'"))
        ]
        if not valid_pos:
            continue
        chosen = rng.sample(valid_pos, min(3, len(valid_pos)))
        for idx in chosen:
            ctx = rec["text"][: matches[idx].start()].strip()
            target = matches[idx].group(0).rstrip(".,!?;:\"'")
            if not ctx or not target:
                continue
            preds = [w.lower() for w, _ in predictor.predict_next_word(ctx, top_k=3)]
            t_low = target.lower()
            rank = preds.index(t_low) + 1 if t_low in preds else 0

            for bucket in (overall, by_source[rec["source"]]):
                bucket["n"] += 1
                if rank == 1:
                    bucket["t1"] += 1
                if 1 <= rank <= 3:
                    bucket["t3"] += 1
                    bucket["mrr"] += 1.0 / rank

    def _fmt(b: dict) -> dict:
        n = b["n"]
        return {
            "n": n,
            "top1_pct": round(b["t1"] / n * 100, 2) if n else 0.0,
            "top3_pct": round(b["t3"] / n * 100, 2) if n else 0.0,
            "mrr": round(b["mrr"] / n, 4) if n else 0.0,
        }

    return {
        "overall": _fmt(overall),
        "by_source": {src: _fmt(st) for src, st in sorted(by_source.items())},
    }


def eval_ksr(
    predictor: FutoPredictor, records: list[dict[str, str]], ksr_sentences: int
) -> dict:
    """Berechnet Keystroke Savings Rate (KSR@1 und KSR@3) beim Wort-für-Wort-Tippen."""
    total_chars = 0
    saved_t1 = 0
    saved_t3 = 0
    words_tested = 0

    for rec in records[:ksr_sentences]:
        matches = list(WORD_RE.finditer(rec["text"]))
        for i, m in enumerate(matches):
            word = m.group(0).rstrip(".,!?;:\"'")
            if not ALPHA_WORD_RE.match(word) or len(word) < 3:
                continue
            ctx = rec["text"][: m.start()].strip()
            ascii_w = _to_ascii_de(word)
            if len(ascii_w) < 3:
                continue

            word_len = len(ascii_w)
            total_chars += word_len
            words_tested += 1
            t_low = word.lower()

            best_k1 = word_len
            best_k3 = word_len

            # 0 getippte Buchstaben (Next-Word-Vorschlag vor dem Tippen)
            if i >= 1:
                nw_preds = [w.lower() for w, _ in predictor.predict_next_word(ctx, top_k=3)]
                if nw_preds and nw_preds[0] == t_low:
                    best_k1 = 0
                    best_k3 = 0
                elif t_low in nw_preds:
                    best_k3 = 0

            # 1..len-1 getippte Buchstaben über <XBU>
            if predictor.supports_xbu and best_k1 > 0:
                for k in range(1, word_len):
                    if k >= best_k1 and k >= best_k3:
                        break
                    xbu_preds = [
                        w.lower()
                        for w, _ in predictor.predict_xbu(ctx, ascii_w[:k], top_k=3)
                    ]
                    if best_k1 > k and xbu_preds and xbu_preds[0] == t_low:
                        best_k1 = k
                    if best_k3 > k and t_low in xbu_preds:
                        best_k3 = k
                    if best_k1 <= k and best_k3 <= k:
                        break

            saved_t1 += max(0, word_len - best_k1)
            saved_t3 += max(0, word_len - best_k3)

    return {
        "sentences": min(ksr_sentences, len(records)),
        "words": words_tested,
        "total_chars": total_chars,
        "ksr_top1_pct": round(saved_t1 / total_chars * 100, 2) if total_chars else 0.0,
        "ksr_top3_pct": round(saved_t3 / total_chars * 100, 2) if total_chars else 0.0,
    }


def eval_xbu_stress(
    predictor: FutoPredictor, records: list[dict[str, str]], rng: random.Random
) -> dict:
    if not predictor.supports_xbu:
        return {"supported": False, "modes": {}}

    modes = ("typo_1", "typo_2", "umlaut", "prefix_typo")
    stats = {
        m: {"n": 0, "t1": 0, "t3": 0, "case1": 0}
        for m in modes
    }

    for rec in records:
        matches = list(WORD_RE.finditer(rec["text"]))
        alpha_positions = [
            (m.start(), m.group(0).rstrip(".,!?;:\"'"))
            for m in matches
            if ALPHA_WORD_RE.match(m.group(0).rstrip(".,!?;:\"'"))
        ]
        if not alpha_positions:
            continue

        # Umlaut-Kandidaten bevorzugt auch testen, falls im Satz vorhanden
        umlaut_cands = [p for p in alpha_positions if any(c in p[1].lower() for c in "äöüß")]
        for mode in modes:
            pool = umlaut_cands if mode == "umlaut" else alpha_positions
            if not pool:
                continue
            start_pos, target = rng.choice(pool)
            corrupted = _corrupt_word(target, mode, rng)
            if not corrupted:
                continue
            ctx = rec["text"][:start_pos].strip()
            preds = [w for w, _ in predictor.predict_xbu(ctx, corrupted, top_k=3)]
            preds_l = [w.lower() for w in preds]
            t_l = target.lower()

            st = stats[mode]
            st["n"] += 1
            if preds_l and preds_l[0] == t_l:
                st["t1"] += 1
            if t_l in preds_l:
                st["t3"] += 1
            if preds and preds[0] == target:
                st["case1"] += 1

    out = {}
    for m, st in stats.items():
        n = st["n"]
        out[m] = {
            "n": n,
            "top1_pct": round(st["t1"] / n * 100, 2) if n else 0.0,
            "top3_pct": round(st["t3"] / n * 100, 2) if n else 0.0,
            "case_top1_pct": round(st["case1"] / n * 100, 2) if n else 0.0,
        }
    return {"supported": True, "modes": out}


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out Next-Word, KSR & XBU-Stress Benchmark.")
    parser.add_argument("model", nargs="?", type=Path, default=None, help="Pfad zu .gguf oder PyTorch-Ordner.")
    parser.add_argument("--input", type=Path, default=config.HELDOUT_PATH)
    parser.add_argument("--max-sentences", type=int, default=210)
    parser.add_argument("--ksr-sentences", type=int, default=70)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    model_path = args.model or resolve_default_model()
    records = load_heldout_records(args.input, args.max_sentences, seed=args.seed)
    if not records:
        raise SystemExit(f"Keine Sätze in {args.input} gefunden.")

    rng = random.Random(args.seed)
    backend = open_backend(model_path, threads=args.threads, device=args.device)
    predictor = FutoPredictor(backend)

    t0 = time.perf_counter()
    try:
        nw_res = eval_next_word(predictor, records, rng)
        ksr_res = eval_ksr(predictor, records, args.ksr_sentences)
        xbu_res = eval_xbu_stress(predictor, records, rng)
    finally:
        backend.close()
    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 74}")
    print(f" HELD-OUT BENCHMARK: {model_path.name} ({len(records)} Sätze, {elapsed:.1f}s)")
    print(f"{'=' * 74}")

    ov = nw_res["overall"]
    print(
        f"\n1) Next-Word ({ov['n']} Positionen): "
        f"Top-1={ov['top1_pct']:.1f}% | Top-3={ov['top3_pct']:.1f}% | MRR={ov['mrr']:.4f}"
    )
    for src, st in nw_res["by_source"].items():
        print(
            f"   - {src:<14} (n={st['n']:>4}): "
            f"Top-1={st['top1_pct']:5.1f}% | Top-3={st['top3_pct']:5.1f}% | MRR={st['mrr']:.4f}"
        )

    print(
        f"\n2) Keystroke Savings Rate ({ksr_res['sentences']} Sätze, {ksr_res['words']} Wörter, "
        f"{ksr_res['total_chars']} Zeichen): "
        f"KSR@Top-1={ksr_res['ksr_top1_pct']:.1f}% | KSR@Top-3={ksr_res['ksr_top3_pct']:.1f}%"
    )

    print("\n3) <XBU>-Stress-Suite (echter Satzkontext):")
    if not xbu_res["supported"]:
        print("   (kein <XBU>-Tokenizer)")
    else:
        for mode, st in xbu_res["modes"].items():
            print(
                f"   - {mode:<14} (n={st['n']:>3}): "
                f"Top-1={st['top1_pct']:5.1f}% | Top-3={st['top3_pct']:5.1f}% | "
                f"Case-T1={st['case_top1_pct']:5.1f}%"
            )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model": str(model_path),
            "heldout_input": str(args.input),
            "sentences": len(records),
            "elapsed_seconds": round(elapsed, 2),
            "next_word": nw_res,
            "ksr": ksr_res,
            "xbu_stress": xbu_res,
        }
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[✓] Held-out-Report gespeichert: {args.json_out}")


if __name__ == "__main__":
    main()
