"""Download-Helper für frei zugängliche v1.5-Trainingsdatenquellen."""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


@dataclass(frozen=True)
class SourceSpec:
    key: str
    description: str
    url: str
    target_rel: str
    license_name: str
    note: str = ""


SOURCES: dict[str, SourceSpec] = {
    "tatoeba": SourceSpec(
        key="tatoeba",
        description="Tatoeba Deutsch (Alltagssätze)",
        url="https://downloads.tatoeba.org/exports/per_language/deu/deu_sentences.tsv.bz2",
        target_rel="tatoeba_deu_sentences.tsv.bz2",
        license_name="CC BY 2.0 FR / CC0",
    ),
    "opensubtitles": SourceSpec(
        key="opensubtitles",
        description="OPUS OpenSubtitles v2018 DE mono (Dialoge)",
        url="https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/de.txt.gz",
        target_rel="opensubtitles_de.txt.gz",
        license_name="Research/Edu (Pflichtlink opensubtitles.org + Lison & Tiedemann 2016)",
    ),
    "leipzig_news": SourceSpec(
        key="leipzig_news",
        description="Leipzig Corpora News DE 1M (2023/2022)",
        url="https://downloads.wortschatz-leipzig.de/corpora/deu_news_2023_1M.tar.gz",
        target_rel="deu_news_2023_1M-sentences.txt",
        license_name="CC BY 4.0 (Uni Leipzig)",
    ),
    "fineweb2": SourceSpec(
        key="fineweb2",
        description="FineWeb-2 deu_Latn (Streaming-Auszug als JSONL.GZ)",
        url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
        target_rel="fineweb2_de.jsonl.gz",
        license_name="ODC-By 1.0",
    ),
}

MANUAL_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="fragreddit",
        description="Reddit r/FragReddit Posts & Comments (Arctic Shift)",
        url="https://arctic-shift.photon-reddit.com/download-utility",
        target_rel="r_fragreddit_posts.jsonl (+ r_fragreddit_comments.jsonl)",
        license_name="Reddit User Agreement (nur lokal, keine Weiterverbreitung)",
    ),
    SourceSpec(
        key="leipzig_web",
        description="Leipzig Corpora Web 2021 DE 1M",
        url="https://wortschatz.uni-leipzig.de/en/download/German",
        target_rel="deu-de_web_2021_1M-sentences.txt",
        license_name="CC BY 4.0 (Uni Leipzig)",
    ),
    SourceSpec(
        key="wikipedia",
        description="Leipzig Corpora Wikipedia 2021 DE 1M",
        url="https://wortschatz.uni-leipzig.de/en/download/German",
        target_rel="deu_wikipedia_2021_1M-sentences.txt",
        license_name="CC BY 4.0 (Uni Leipzig / Wikipedia CC BY-SA)",
    ),
)

LEIPZIG_NEWS_CANDIDATES = (
    (
        "https://downloads.wortschatz-leipzig.de/corpora/deu_news_2023_1M.tar.gz",
        "deu_news_2023_1M-sentences.txt",
    ),
    (
        "https://downloads.wortschatz-leipzig.de/corpora/deu_news_2022_1M.tar.gz",
        "deu_news_2022_1M-sentences.txt",
    ),
)


def _format_size(path: Path) -> str:
    size = path.stat().st_size
    if size >= 1024 ** 3:
        return f"{size / (1024 ** 3):.2f} GB"
    if size >= 1024 ** 2:
        return f"{size / (1024 ** 2):.1f} MB"
    return f"{size / 1024:.1f} KB"


def _download_file(url: str, dest: Path, desc: str) -> None:
    """Lädt eine Datei atomar via <dest>.part mit Fortschrittsbalken herunter."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            part.open("wb") as f,
            tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
            ) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    part.replace(dest)


def download_simple(spec: SourceSpec, dataset_dir: Path, force: bool) -> None:
    dest = dataset_dir / spec.target_rel
    if dest.exists() and not force:
        print(f"[übersprungen] {spec.key}: {dest} existiert bereits ({_format_size(dest)})")
        return
    print(f"[download] {spec.key}: {spec.url} -> {dest}")
    _download_file(spec.url, dest, spec.key)
    print(f"[✓] {spec.key} gespeichert: {dest} ({_format_size(dest)})")


def download_leipzig_news(dataset_dir: Path, force: bool) -> None:
    existing = sorted(dataset_dir.glob("deu_news_*_1M-sentences.txt"))
    if existing and not force:
        print(f"[übersprungen] leipzig_news: {existing[-1]} existiert bereits ({_format_size(existing[-1])})")
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url, target_name in LEIPZIG_NEWS_CANDIDATES:
        archive_path = dataset_dir / f"{target_name}.tar.gz"
        dest_path = dataset_dir / target_name
        try:
            print(f"[download] leipzig_news: {url}")
            _download_file(url, archive_path, "leipzig_news")
            extracted = False
            with tarfile.open(archive_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if member.isfile() and member.name.endswith("-sentences.txt"):
                        src_f = tf.extractfile(member)
                        if src_f is None:
                            continue
                        part = dest_path.with_name(dest_path.name + ".part")
                        with src_f, part.open("wb") as out_f:
                            while True:
                                buf = src_f.read(1024 * 512)
                                if not buf:
                                    break
                                out_f.write(buf)
                        part.replace(dest_path)
                        extracted = True
                        break
            archive_path.unlink(missing_ok=True)
            if not extracted:
                raise RuntimeError(f"Keine *-sentences.txt im Archiv gefunden: {url}")
            print(f"[✓] leipzig_news entpackt: {dest_path} ({_format_size(dest_path)})")
            return
        except Exception as exc:
            archive_path.unlink(missing_ok=True)
            last_err = exc
            print(f"[hinweis] Fallback für leipzig_news nach Fehler bei {url}: {exc}")

    raise RuntimeError(f"Leipzig News Download fehlgeschlagen: {last_err}")


def download_fineweb2(
    dataset_dir: Path, max_mb: int, min_chars: int, force: bool
) -> None:
    from datasets import load_dataset

    dest = dataset_dir / SOURCES["fineweb2"].target_rel
    if dest.exists() and not force:
        print(f"[übersprungen] fineweb2: {dest} existiert bereits ({_format_size(dest)})")
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = max_mb * 1024 * 1024
    written_bytes = 0
    docs = 0
    part = dest.with_name(dest.name + ".part")

    print(f"[stream] FineWeb-2 (deu_Latn), Ziel: {max_mb} MB Rohtext -> {dest}")
    stream = load_dataset(
        "HuggingFaceFW/fineweb-2", name="deu_Latn", split="train", streaming=True
    )
    with (
        part.open("wb") as raw_f,
        gzip.GzipFile(
            filename="fineweb2_de.jsonl", mode="wb", fileobj=raw_f, compresslevel=5
        ) as gz_bin,
        io.TextIOWrapper(gz_bin, encoding="utf-8") as gz,
        tqdm(total=max_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc="fineweb2") as pbar,
    ):
        for row in stream:
            text = (row.get("text") or "").strip()
            if len(text) < min_chars:
                continue
            gz.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            delta = len(text.encode("utf-8"))
            written_bytes += delta
            docs += 1
            pbar.update(min(delta, max(0, max_bytes - pbar.n)))
            if written_bytes >= max_bytes:
                break

    part.replace(dest)
    print(
        f"[✓] fineweb2 gespeichert: {dest} ({_format_size(dest)} komprimiert, "
        f"{docs:,} Dokumente, {written_bytes / (1024 ** 2):.1f} MB Rohtext)"
    )


def print_source_list(dataset_dir: Path) -> None:
    print(f"=== Konfigurierte Datenquellen ({dataset_dir}) ===\n")
    print("[Automatisch per download_datasets.py]")
    for spec in SOURCES.values():
        if spec.key == "leipzig_news":
            matches = sorted(dataset_dir.glob("deu_news_*_1M-sentences.txt"))
            path = matches[-1] if matches else (dataset_dir / spec.target_rel)
        else:
            path = dataset_dir / spec.target_rel
        status = f"VORHANDEN ({_format_size(path)})" if path.exists() else "fehlt"
        print(f"  - {spec.key:<14} | {status:<22} | Lizenz: {spec.license_name}")
        print(f"    Beschreibung: {spec.description}")
        print(f"    Datei:        {path}")
        print(f"    URL:          {spec.url}")

    print("\n[Manuell abzulegende Quellen]")
    for spec in MANUAL_SOURCES:
        if spec.key == "fragreddit":
            posts = dataset_dir / "r_fragreddit_posts.jsonl"
            comments = dataset_dir / "r_fragreddit_comments.jsonl"
            if posts.exists() and comments.exists():
                status = f"VORHANDEN ({_format_size(posts)} + {_format_size(comments)})"
            elif posts.exists() or comments.exists():
                status = "TEILWEISE VORHANDEN"
            else:
                status = "fehlt"
        else:
            path = dataset_dir / spec.target_rel
            status = f"VORHANDEN ({_format_size(path)})" if path.exists() else "fehlt"
        print(f"  - {spec.key:<14} | {status:<28} | Lizenz: {spec.license_name}")
        print(f"    Beschreibung: {spec.description}")
        print(f"    Datei(en):    {dataset_dir / spec.target_rel}")
        print(f"    URL:          {spec.url}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lädt freie deutsche Trainingsdatenquellen in DataSet/ herunter."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=[*SOURCES.keys(), "all"],
        default=["all"],
        help="Welche Quellen geladen werden sollen (Standard: all).",
    )
    parser.add_argument("--dataset-dir", type=Path, default=config.DATASET_DIR)
    parser.add_argument(
        "--fineweb-max-mb",
        type=int,
        default=600,
        help="Unkomprimiertes Textlimit für FineWeb-2 deu_Latn Streaming in MB.",
    )
    parser.add_argument("--fineweb-min-chars", type=int, default=120)
    parser.add_argument("--force", action="store_true", help="Existierende Dateien überschreiben.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Alle Quellen, URLs, Lizenzen und lokalen Status anzeigen (kein Download).",
    )
    args = parser.parse_args()

    if args.list:
        print_source_list(args.dataset_dir)
        return

    selected = list(SOURCES.keys()) if "all" in args.sources else list(dict.fromkeys(args.sources))
    for key in selected:
        if key in {"tatoeba", "opensubtitles"}:
            download_simple(SOURCES[key], args.dataset_dir, force=args.force)
        elif key == "leipzig_news":
            download_leipzig_news(args.dataset_dir, force=args.force)
        elif key == "fineweb2":
            download_fineweb2(
                args.dataset_dir,
                max_mb=args.fineweb_max_mb,
                min_chars=args.fineweb_min_chars,
                force=args.force,
            )


if __name__ == "__main__":
    main()
