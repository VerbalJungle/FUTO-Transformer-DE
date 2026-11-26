"""Erstellt den bereinigten v1.5 Multi-Source-Tastatur-Korpus inkl. Held-out-Split & Cooldown."""
from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import html
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\xad]")
QUOTE_LINE_RE = re.compile(r"(?m)^[ \t]*>.*$")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.|/)[^\)]+\)")
URL_MENTION_RE = re.compile(
    r"https?://\S+|www\.\S+|/?[ur]/[A-Za-z0-9_-]+|@\w+|#\w+"
)
SPOILER_RE = re.compile(r">!|!<")
MD_FORMAT_RE = re.compile(r"[*_~`]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_TOKEN_RE = re.compile(r"[a-zäöüß]+")

# Untertitel-spezifische Muster (OpenSubtitles)
SUB_STAGE_DIR_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
SUB_MUSIC_RE = re.compile(r"[♪♫]+")
SUB_LEAD_DASH_RE = re.compile(r"^\s*[-–—]+\s*")
SUB_SPEAKER_RE = re.compile(r"^[A-ZÄÖÜ][A-ZÄÖÜ0-9 _-]{1,20}:\s*")
SUB_CREDIT_WORDS = ("untertitel", "übersetzung", "opensubtitles", "sync by", "www.")

GARBAGE_PREFIXES = (";", ")", "]", "|", "$", "=", "/", "\\", ",", ".")

DE_STOPWORDS = frozenset({
    "ich", "du", "er", "sie", "es", "wir", "ihr", "der", "die", "das", "ein", "eine",
    "und", "oder", "aber", "nicht", "auch", "mit", "für", "auf", "ist", "sind", "war",
    "haben", "hat", "kann", "wie", "was", "wenn", "weil", "dass", "man", "noch", "schon",
    "mal", "mir", "mich", "dir", "dich", "sich", "bei", "von", "zu", "zum", "zur",
})
EN_ONLY_STOPWORDS = frozenset({
    "the", "and", "that", "have", "for", "not", "with", "you", "this", "but", "his",
    "from", "they", "would", "there", "their", "what", "about", "which", "when",
    "your", "just", "should", "could", "because", "really", "actually", "though",
})

COOLDOWN_SOURCES = frozenset({"tatoeba", "opensubtitles", "fragreddit"})


def is_english_only(text: str) -> bool:
    """Filtert rein englische Kommentare anhand typischer Funktionswörter."""
    words = WORD_TOKEN_RE.findall(text.lower())
    if len(words) < 4:
        return False
    en_hits = sum(1 for w in words if w in EN_ONLY_STOPWORDS)
    de_hits = sum(1 for w in words if w in DE_STOPWORDS)
    return en_hits >= 3 and de_hits == 0


def clean_fragment(text: str, min_words: int = 2, max_words: int = 80) -> str | None:
    """Bereinigt ein Textfragment und prüft Länge, Zeichenverteilung und Sprache."""
    if not text or not isinstance(text, str):
        return None
    text = html.unescape(html.unescape(text))
    text = ZERO_WIDTH_RE.sub("", text)
    text = QUOTE_LINE_RE.sub("", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = URL_MENTION_RE.sub("", text)
    text = SPOILER_RE.sub("", text)
    text = MD_FORMAT_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    text = re.sub(r"([!?])\1{2,}", r"\1\1", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"[ \t]+", " ", text).strip()

    if not text or text[0] in GARBAGE_PREFIXES:
        return None
    words = text.split()
    if not (min_words <= len(words) <= max_words):
        return None
    alpha_space = sum(1 for c in text if c.isalpha() or c == " ")
    if alpha_space / len(text) < 0.65:
        return None
    if is_english_only(text):
        return None
    return text


def clean_subtitle_line(line: str) -> str | None:
    """Bereinigt eine einzelne OpenSubtitles-Zeile von Regieanweisungen und Credits."""
    if not line:
        return None
    low = line.lower()
    if any(cw in low for cw in SUB_CREDIT_WORDS):
        return None
    text = SUB_MUSIC_RE.sub("", line)
    text = SUB_STAGE_DIR_RE.sub("", text)
    text = SUB_LEAD_DASH_RE.sub("", text)
    text = SUB_SPEAKER_RE.sub("", text)
    return clean_fragment(text, min_words=2, max_words=32)


def split_into_chunks(text: str, max_words: int = 80) -> list[str]:
    """Behält kurze Mehr-Satz-Absätze zusammen und teilt lange Absätze an Satzgrenzen."""
    if not text or not isinstance(text, str):
        return []
    chunks: list[str] = []
    for raw_para in re.split(r"\n+", text):
        cleaned_para = clean_fragment(raw_para, min_words=2, max_words=max_words)
        if cleaned_para is not None:
            chunks.append(cleaned_para)
            continue

        para = html.unescape(html.unescape(ZERO_WIDTH_RE.sub("", raw_para))).strip()
        if not para or len(para.split()) <= max_words:
            continue
        buf: list[str] = []
        buf_words = 0
        for sent in SENTENCE_SPLIT_RE.split(para):
            s_words = len(sent.split())
            if buf and buf_words + s_words > max_words:
                joined = clean_fragment(" ".join(buf), min_words=2, max_words=max_words)
                if joined:
                    chunks.append(joined)
                buf = [sent]
                buf_words = s_words
            else:
                buf.append(sent)
                buf_words += s_words
        if buf:
            joined = clean_fragment(" ".join(buf), min_words=2, max_words=max_words)
            if joined:
                chunks.append(joined)
    return chunks


def _text_hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def _open_text_auto(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def load_fragreddit(
    dataset_dir: Path, min_score: int, max_lines: int, seen: set[bytes]
) -> list[str]:
    """Lädt Posts und Kommentare aus r_fragreddit_*.jsonl mit Qualitätsfilter."""
    files = [
        dataset_dir / "r_fragreddit_posts.jsonl",
        dataset_dir / "r_fragreddit_comments.jsonl",
    ]
    results: list[str] = []
    for path in files:
        if not path.exists():
            print(f"  (Überspringe fehlende Datei: {path})")
            continue
        file_chunks = 0
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f, start=1):
                if max_lines > 0 and line_idx > max_lines:
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                author = str(data.get("author") or "")
                if author in {"AutoModerator", "[deleted]"} or author.lower().endswith("bot"):
                    continue

                score = data.get("score")
                if isinstance(score, (int, float)) and score < min_score:
                    continue

                for seg in (data.get("title", ""), data.get("selftext", ""), data.get("body", "")):
                    if not seg or seg in {"[deleted]", "[removed]"}:
                        continue
                    if "automoderator" in seg.lower():
                        continue
                    for chunk in split_into_chunks(seg):
                        h = _text_hash(chunk)
                        if h not in seen:
                            seen.add(h)
                            results.append(chunk)
                            file_chunks += 1
        print(f"  -> {file_chunks:,} Chunks aus {path.name}")
    return results


def load_leipzig_tsv(path: Path | None, max_lines: int, seen: set[bytes]) -> list[str]:
    """Lädt bereinigte Sätze aus einer Leipzig-Corpora-*-sentences.txt Datei."""
    if path is None or not path.exists():
        print(f"  (Überspringe fehlende Leipzig-Datei: {path})")
        return []
    results: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, start=1):
            if max_lines > 0 and idx > max_lines:
                break
            parts = line.strip().split("\t", 1)
            raw = parts[1] if len(parts) > 1 else parts[0]
            cleaned = clean_fragment(raw, min_words=2, max_words=80)
            if cleaned is None:
                continue
            h = _text_hash(cleaned)
            if h not in seen:
                seen.add(h)
                results.append(cleaned)
    print(f"  -> {len(results):,} Sätze aus {path.name}")
    return results


def load_tatoeba(dataset_dir: Path, max_lines: int, seen: set[bytes]) -> list[str]:
    """Lädt deutsche Alltagssätze aus tatoeba_deu_sentences.tsv[.bz2]."""
    path = next(
        (
            p
            for p in (
                dataset_dir / "tatoeba_deu_sentences.tsv.bz2",
                dataset_dir / "tatoeba_deu_sentences.tsv",
                dataset_dir / "deu_sentences.tsv.bz2",
            )
            if p.exists()
        ),
        None,
    )
    if path is None:
        print(f"  (Überspringe fehlende Datei: {dataset_dir / 'tatoeba_deu_sentences.tsv.bz2'})")
        return []
    results: list[str] = []
    with _open_text_auto(path) as f:
        for idx, line in enumerate(f, start=1):
            if max_lines > 0 and idx > max_lines:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[1] == "deu":
                raw = parts[2]
            elif len(parts) == 1:
                raw = parts[0]
            else:
                continue
            cleaned = clean_fragment(raw, min_words=2, max_words=60)
            if cleaned is None:
                continue
            h = _text_hash(cleaned)
            if h not in seen:
                seen.add(h)
                results.append(cleaned)
    print(f"  -> {len(results):,} Sätze aus {path.name}")
    return results


def load_opensubtitles(dataset_dir: Path, max_lines: int, seen: set[bytes]) -> list[str]:
    """Lädt OpenSubtitles DE mit Regie-Filter und Dialog-Turn-Joining (max. 3 Turns / 32 Wörter)."""
    path = next(
        (
            p
            for p in (
                dataset_dir / "opensubtitles_de.txt.gz",
                dataset_dir / "opensubtitles_de.txt",
            )
            if p.exists()
        ),
        None,
    )
    if path is None:
        print(f"  (Überspringe fehlende Datei: {dataset_dir / 'opensubtitles_de.txt.gz'})")
        return []

    results: list[str] = []
    buf: list[str] = []
    buf_words = 0

    def flush_buf() -> None:
        nonlocal buf, buf_words
        if not buf:
            return
        joined = " ".join(buf)
        h = _text_hash(joined)
        if h not in seen:
            seen.add(h)
            results.append(joined)
        buf = []
        buf_words = 0

    with _open_text_auto(path) as f:
        for idx, line in enumerate(f, start=1):
            if max_lines > 0 and idx > max_lines:
                break
            cleaned = clean_subtitle_line(line)
            if cleaned is None:
                flush_buf()
                continue
            w_cnt = len(cleaned.split())
            if buf and (len(buf) >= 3 or buf_words + w_cnt > 32):
                flush_buf()
            buf.append(cleaned)
            buf_words += w_cnt
            # Abgeschlossene längere Äußerung direkt flushen, kurze Turns koppeln
            if buf_words >= 16 and cleaned.endswith((".", "!", "?")):
                flush_buf()
    flush_buf()
    print(f"  -> {len(results):,} Dialog-Chunks aus {path.name}")
    return results


def load_fineweb2(dataset_dir: Path, max_lines: int, seen: set[bytes]) -> list[str]:
    """Lädt FineWeb-2 DE (fineweb2_de.jsonl[.gz]) und teilt Absätze sauber auf."""
    path = next(
        (
            p
            for p in (
                dataset_dir / "fineweb2_de.jsonl.gz",
                dataset_dir / "fineweb2_de.jsonl",
            )
            if p.exists()
        ),
        None,
    )
    if path is None:
        print(f"  (Überspringe fehlende Datei: {dataset_dir / 'fineweb2_de.jsonl.gz'})")
        return []
    results: list[str] = []
    with _open_text_auto(path) as f:
        for idx, line in enumerate(f, start=1):
            if max_lines > 0 and idx > max_lines:
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = data.get("text", "")
            for chunk in split_into_chunks(text, max_words=80):
                h = _text_hash(chunk)
                if h not in seen:
                    seen.add(h)
                    results.append(chunk)
    print(f"  -> {len(results):,} Chunks aus {path.name}")
    return results


def _preload_heldout(heldout_path: Path, seen: set[bytes]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if not heldout_path.exists():
        return records
    with heldout_path.open("r", encoding="utf-8") as f:
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
                source = "unknown"
            if text:
                seen.add(_text_hash(text))
                records.append({"source": source, "text": text})
    return records


def _extract_heldout_split(
    pools: dict[str, list[str]],
    heldout_path: Path,
    heldout_per_source: int,
    rebuild: bool,
    existing_records: list[dict[str, str]],
) -> list[dict[str, str]]:
    if existing_records and not rebuild:
        print(
            f"[✓] Bestehendes Held-out-Set behalten ({len(existing_records):,} Sätze, Hashes aus Training blockiert)"
        )
        return existing_records

    rng = random.Random(1337)
    records: list[dict[str, str]] = []
    for source, chunks in pools.items():
        if not chunks or heldout_per_source <= 0:
            continue
        eligible = [
            i for i, txt in enumerate(chunks) if 4 <= len(txt.split()) <= 35
        ]
        cap = min(heldout_per_source, len(chunks) // 10, len(eligible))
        if cap <= 0:
            continue
        picked_indices = set(rng.sample(eligible, cap))
        remaining: list[str] = []
        for idx, txt in enumerate(chunks):
            if idx in picked_indices:
                records.append({"source": source, "text": txt})
            else:
                remaining.append(txt)
        pools[source] = remaining

    heldout_path.parent.mkdir(parents=True, exist_ok=True)
    with heldout_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[✓] Held-out-Benchmark gespeichert: {heldout_path} ({len(records):,} Sätze)")
    return records


def _mix_proportional(
    pools: dict[str, list[str]],
    weights: dict[str, float],
    max_total_lines: int,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Mischt Quellen ohne Oversampling möglichst exakt gemäß Zielgewichten."""
    active = {k: v for k, v in pools.items() if v and weights.get(k, 0.0) > 0.0}
    if not active:
        return {}

    rng = random.Random(seed)
    total_w = sum(weights[k] for k in active)
    norm_w = {k: weights[k] / total_w for k in active}

    # Iterativ kleine erschöpfte Quellen mit 100 % einbeziehen, restliches Budget nach Gewicht verteilen
    fixed_counts: dict[str, int] = {}
    remaining = dict(norm_w)

    # Obergrenze, wenn --max-total-lines gesetzt ist, sonst unbegrenzt
    target_total = max_total_lines if max_total_lines > 0 else sum(len(pools[k]) for k in active)
    target_total = min(target_total, sum(len(pools[k]) for k in active))

    while remaining:
        rem_w_sum = sum(remaining.values())
        rem_budget = target_total - sum(fixed_counts.values())
        if rem_budget <= 0:
            for k in list(remaining):
                fixed_counts[k] = 0
            break

        # Prüfe ob eine Quelle weniger Daten hat als ihr proportionales Budget (bei max_total_lines == 0:
        # wähle größte gemeinsame Skalierung so, dass Hauptquellen ihr Verhältnis halten,
        # oder behalte kleine Quellen vollständig solange keine große Quelle < 25% ihrer Daten fällt)
        if max_total_lines > 0:
            bottlenecks = [
                k for k, w in remaining.items() if len(pools[k]) < int(rem_budget * (w / rem_w_sum))
            ]
            if bottlenecks:
                for k in bottlenecks:
                    fixed_counts[k] = len(pools[k])
                    del remaining[k]
                continue
            for k, w in remaining.items():
                fixed_counts[k] = min(len(pools[k]), int(rem_budget * (w / rem_w_sum)))
            break
        else:
            # Ohne explizites Limit: bestimme Skalierung S aus nicht-winzigen Quellen (>= 5% Zielgewicht)
            # oder maximal möglichem Budget ohne Oversampling
            major = [k for k, w in remaining.items() if norm_w[k] >= 0.05] or list(remaining)
            scale = min(len(pools[k]) / (remaining[k] / rem_w_sum) for k in major)
            exhausted_minor = [
                k for k in remaining if len(pools[k]) < int(scale * (remaining[k] / rem_w_sum))
            ]
            if exhausted_minor:
                for k in exhausted_minor:
                    fixed_counts[k] = len(pools[k])
                    del remaining[k]
                continue
            for k, w in remaining.items():
                fixed_counts[k] = min(len(pools[k]), int(scale * (w / rem_w_sum)))
            break

    selected: dict[str, list[str]] = {}
    for k, cnt in fixed_counts.items():
        if cnt <= 0:
            selected[k] = []
        elif cnt >= len(pools[k]):
            selected[k] = list(pools[k])
        else:
            selected[k] = rng.sample(pools[k], cnt)
    return selected


def build_corpus(
    output_path: Path = config.CORPUS_PATH,
    dataset_dir: Path = config.DATASET_DIR,
    min_reddit_score: int = 1,
    max_reddit_lines: int = 0,
    max_wiki_lines: int = 0,
    max_leipzig_web_lines: int = 0,
    max_per_source: int = 0,
    max_total_lines: int = 0,
    heldout_output: Path | None = config.HELDOUT_PATH,
    heldout_per_source: int = 300,
    rebuild_heldout: bool = False,
    cooldown_output: Path | None = config.COOLDOWN_CORPUS_PATH,
    cooldown_max_lines: int = 300_000,
    stats_output: Path | None = config.CORPUS_STATS_PATH,
) -> Path:
    seen: set[bytes] = set()
    existing_heldout: list[dict[str, str]] = []
    if heldout_output is not None and not rebuild_heldout:
        existing_heldout = _preload_heldout(heldout_output, seen)
        if existing_heldout:
            print(
                f"=== 0. Held-out Vorab-Blockierung: {len(existing_heldout):,} Sätze aus {heldout_output.name} ==="
            )

    def _cap(specific: int) -> int:
        if specific > 0 and max_per_source > 0:
            return min(specific, max_per_source)
        return specific or max_per_source

    news_files = sorted(dataset_dir.glob("deu_news_*_1M-sentences.txt"))
    news_path = news_files[-1] if news_files else None

    pools: dict[str, list[str]] = {}
    print("=== 1. Lade FragReddit ===")
    pools["fragreddit"] = load_fragreddit(dataset_dir, min_reddit_score, _cap(max_reddit_lines), seen)

    print("=== 2. Lade OpenSubtitles DE ===")
    pools["opensubtitles"] = load_opensubtitles(dataset_dir, max_per_source, seen)

    print("=== 3. Lade FineWeb-2 DE ===")
    pools["fineweb2"] = load_fineweb2(dataset_dir, max_per_source, seen)

    print("=== 4. Lade Leipzig Web 2021 ===")
    pools["leipzig_web"] = load_leipzig_tsv(
        dataset_dir / "deu-de_web_2021_1M-sentences.txt", _cap(max_leipzig_web_lines), seen
    )

    print("=== 5. Lade Leipzig Wikipedia 2021 ===")
    pools["wikipedia"] = load_leipzig_tsv(
        dataset_dir / "deu_wikipedia_2021_1M-sentences.txt", _cap(max_wiki_lines), seen
    )

    print("=== 6. Lade Leipzig News ===")
    pools["leipzig_news"] = load_leipzig_tsv(news_path, max_per_source, seen)

    print("=== 7. Lade Tatoeba DE ===")
    pools["tatoeba"] = load_tatoeba(dataset_dir, max_per_source, seen)

    heldout_records: list[dict[str, str]] = []
    if heldout_output is not None:
        heldout_records = _extract_heldout_split(
            pools, heldout_output, heldout_per_source, rebuild_heldout, existing_heldout
        )

    selected = _mix_proportional(pools, config.CORPUS_SOURCE_WEIGHTS, max_total_lines=max_total_lines)
    if not selected:
        raise RuntimeError(f"Keine Quelldaten in {dataset_dir} gefunden.")

    combined: list[str] = []
    cooldown_pool: list[str] = []
    for src, chunks in selected.items():
        combined.extend(chunks)
        if src in COOLDOWN_SOURCES:
            for ch in chunks:
                if 3 <= len(ch.split()) <= 28:
                    cooldown_pool.append(ch)

    rng = random.Random(42)
    rng.shuffle(combined)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in combined:
            f.write(entry + "\n")

    cooldown_written = 0
    if cooldown_output is not None and cooldown_pool:
        rng.shuffle(cooldown_pool)
        if cooldown_max_lines > 0 and len(cooldown_pool) > cooldown_max_lines:
            cooldown_pool = cooldown_pool[:cooldown_max_lines]
        cooldown_output.parent.mkdir(parents=True, exist_ok=True)
        with cooldown_output.open("w", encoding="utf-8") as f:
            for entry in cooldown_pool:
                f.write(entry + "\n")
        cooldown_written = len(cooldown_pool)

    total_lines = len(combined)
    total_words = sum(len(line.split()) for line in combined)
    size_mb = output_path.stat().st_size / (1024 * 1024)

    print("\n=== Korpus-Mischung (v1.5) ===")
    source_stats: dict[str, dict] = {}
    for src, chunks in selected.items():
        cnt = len(chunks)
        words = sum(len(c.split()) for c in chunks)
        pct = (cnt / total_lines * 100.0) if total_lines else 0.0
        source_stats[src] = {
            "pool_filtered": len(pools.get(src, [])),
            "selected_lines": cnt,
            "selected_words": words,
            "share_percent": round(pct, 2),
        }
        print(f"  - {src:<14}: {cnt:>9,} Zeilen ({pct:5.1f}%) | Pool: {len(pools.get(src, [])):,}")

    print(
        f"\n[✓] Fertig: {total_lines:,} Zeilen | {total_words:,} Wörter | "
        f"{size_mb:.1f} MB -> {output_path}"
    )
    if cooldown_written:
        print(f"[✓] Cooldown-Korpus: {cooldown_written:,} Zeilen -> {cooldown_output}")

    if stats_output is not None:
        stats_output.parent.mkdir(parents=True, exist_ok=True)
        stats_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "corpus_path": str(output_path),
            "size_mb": round(size_mb, 2),
            "total_lines": total_lines,
            "total_words": total_words,
            "heldout_path": str(heldout_output) if heldout_output else None,
            "heldout_lines": len(heldout_records),
            "cooldown_path": str(cooldown_output) if cooldown_written else None,
            "cooldown_lines": cooldown_written,
            "sources": source_stats,
        }
        stats_output.write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[✓] Statistik gespeichert: {stats_output}")

    return output_path


def _optional_path(raw: str) -> Path | None:
    if not raw or raw.lower() == "none":
        return None
    return Path(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Erstellt den FUTO-DE v1.5 Multi-Source-Korpus.")
    parser.add_argument("--output", type=Path, default=config.CORPUS_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--min-reddit-score", type=int, default=1)
    parser.add_argument("--max-reddit-lines", type=int, default=0)
    parser.add_argument("--max-wiki-lines", type=int, default=0)
    parser.add_argument("--max-leipzig-web-lines", type=int, default=0)
    parser.add_argument("--max-per-source", type=int, default=0, help="Globales Rohzeilen-Limit je Quelle (Tests).")
    parser.add_argument("--max-total-lines", type=int, default=0, help="Maximale Zeilen im finalen Mix (0 = unbegrenzt).")
    parser.add_argument("--heldout-output", type=str, default=str(config.HELDOUT_PATH))
    parser.add_argument("--heldout-per-source", type=int, default=300)
    parser.add_argument("--rebuild-heldout", action="store_true")
    parser.add_argument("--cooldown-output", type=str, default=str(config.COOLDOWN_CORPUS_PATH))
    parser.add_argument("--cooldown-max-lines", type=int, default=300_000)
    parser.add_argument("--stats-output", type=str, default=str(config.CORPUS_STATS_PATH))
    args = parser.parse_args()

    build_corpus(
        output_path=args.output,
        dataset_dir=args.dataset_dir,
        min_reddit_score=args.min_reddit_score,
        max_reddit_lines=args.max_reddit_lines,
        max_wiki_lines=args.max_wiki_lines,
        max_leipzig_web_lines=args.max_leipzig_web_lines,
        max_per_source=args.max_per_source,
        max_total_lines=args.max_total_lines,
        heldout_output=_optional_path(args.heldout_output),
        heldout_per_source=args.heldout_per_source,
        rebuild_heldout=args.rebuild_heldout,
        cooldown_output=_optional_path(args.cooldown_output),
        cooldown_max_lines=args.cooldown_max_lines,
        stats_output=_optional_path(args.stats_output),
    )


if __name__ == "__main__":
    main()
