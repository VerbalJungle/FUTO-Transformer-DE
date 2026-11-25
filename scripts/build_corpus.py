"""Erstellt den bereinigten deutschen Tastatur-Korpus (ca. 85% FragReddit, 15% Wikipedia 2021)."""
import argparse
import hashlib
import html
import json
import random
import re
import sys
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

        # Absatz war zu lang (> max_words) oder enthielt Rohtext -> an Satzgrenzen gruppieren
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


def load_wikipedia(dataset_dir: Path, max_lines: int, seen: set[bytes]) -> list[str]:
    """Lädt bereinigte Sätze aus deu_wikipedia_2021_1M-sentences.txt."""
    path = dataset_dir / "deu_wikipedia_2021_1M-sentences.txt"
    if not path.exists():
        print(f"  (Überspringe fehlende Datei: {path})")
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


def build_corpus(
    output_path: Path = config.CORPUS_PATH,
    wiki_ratio: float = 0.15,
    min_reddit_score: int = 1,
    max_reddit_lines: int = 0,
    max_wiki_lines: int = 0,
) -> Path:
    if not (0.0 <= wiki_ratio < 1.0):
        raise ValueError(f"wiki_ratio muss in [0.0, 1.0) liegen, ist {wiki_ratio}")

    seen: set[bytes] = set()
    print("=== 1. Lade FragReddit ===")
    reddit_chunks = load_fragreddit(config.DATASET_DIR, min_reddit_score, max_reddit_lines, seen)

    print("=== 2. Lade Wikipedia 2021 ===")
    wiki_chunks = load_wikipedia(config.DATASET_DIR, max_wiki_lines, seen)

    rng = random.Random(42)
    if wiki_ratio == 0.0:
        wiki_selected = []
    elif reddit_chunks:
        max_wiki = int(len(reddit_chunks) * wiki_ratio / (1.0 - wiki_ratio))
        if len(wiki_chunks) > max_wiki:
            wiki_selected = rng.sample(wiki_chunks, max_wiki)
        else:
            wiki_selected = wiki_chunks
    else:
        wiki_selected = wiki_chunks

    combined = reddit_chunks + wiki_selected
    rng.shuffle(combined)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in combined:
            f.write(entry + "\n")

    total = len(combined)
    actual_wiki_pct = (len(wiki_selected) / total * 100) if total else 0.0
    print(
        f"=== Fertig: {total:,} Zeilen -> {output_path} "
        f"(Reddit: {len(reddit_chunks):,}, Wiki: {len(wiki_selected):,} / {actual_wiki_pct:.1f}%) ==="
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Erstellt den FUTO-DE Korpus.")
    parser.add_argument("--output", type=Path, default=config.CORPUS_PATH)
    parser.add_argument("--wiki-ratio", type=float, default=0.15)
    parser.add_argument("--min-reddit-score", type=int, default=1)
    parser.add_argument("--max-reddit-lines", type=int, default=0)
    parser.add_argument("--max-wiki-lines", type=int, default=0)
    args = parser.parse_args()
    build_corpus(
        output_path=args.output,
        wiki_ratio=args.wiki_ratio,
        min_reddit_score=args.min_reddit_score,
        max_reddit_lines=args.max_reddit_lines,
        max_wiki_lines=args.max_wiki_lines,
    )


if __name__ == "__main__":
    main()
