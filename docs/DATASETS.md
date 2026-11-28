# Trainingsdatenquellen & Korpus-Dokumentation (FUTO-DE)

Dieses Dokument beschreibt alle verwendeten Rohdatenquellen für den deutschen FUTO-Keyboard-Korpus, inklusive Download-Links, erwarteten Dateinamen in `DataSet/`, Zielanteilen für **v1.5**, Lizenzbedingungen und Filterregeln.

---

## 1. Übersicht der Datenquellen (v1.5)

| Quelle | Lokale Datei(en) in `DataSet/` | Domäne / Zweck | Zielanteil v1.5 | Lizenz / Bedingungen | Bezugsquelle |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Reddit `r/FragReddit`** *(via Arctic Shift)* | `r_fragreddit_posts.jsonl`<br>`r_fragreddit_comments.jsonl` | Alltagssprache, Fragen & Antworten, lockeres Forum-Deutsch | ~32 % | Reddit User Agreement / Urheberrechte der jeweiligen Autoren *(nur lokal beziehen, **keine Weiterverbreitung** des Rohkorpus)* | [Arctic Shift Download Utility](https://arctic-shift.photon-reddit.com/download-utility) |
| **OPUS OpenSubtitles (DE, v2018 mono)** | `opensubtitles_de.txt.gz` | Gesprochene Dialoge & kurze Chat-Antworten | ~23 % | Forschungs- & Bildungsnutzung; **Pflicht-Attribution** (Link auf `opensubtitles.org` + Zitation Lison & Tiedemann 2016) | [Direktdownload `de.txt.gz`](https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/de.txt.gz) / [OPUS Portal](https://opus.nlpl.eu/OpenSubtitles-v2018.php) |
| **FineWeb-2 (`HuggingFaceFW/fineweb-2`, `deu_Latn`)** | `fineweb2_de.jsonl.gz` *(Streaming-Auszug, Standard ~600 MB Text)* | Web, Blogs, Ratgeber, breite Alltagsthemen | ~22 % | **ODC-By 1.0** (Open Data Commons Attribution; kommerziell nutzbar mit Namensnennung) | [HuggingFace `HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) |
| **Leipzig Corpora Collection – Web 2021** (`deu-de_web_2021_1M`) | `deu-de_web_2021_1M-sentences.txt` | Allgemeine deutsche Web-Sätze | ~9 % | **CC BY 4.0** (Universität Leipzig / Wortschatz) | [Wortschatz Leipzig Downloads](https://wortschatz.uni-leipzig.de/en/download/German) |
| **Leipzig Corpora Collection – Wikipedia 2021** (`deu_wikipedia_2021_1M`) | `deu_wikipedia_2021_1M-sentences.txt` | Enzyklopädie, saubere Rechtschreibung & Entitäten | ~6 % | **CC BY 4.0** (Universität Leipzig / Wikipedia CC BY-SA) | [Wortschatz Leipzig Downloads](https://wortschatz.uni-leipzig.de/en/download/German) |
| **Leipzig Corpora Collection – News** (`deu_news_2023_1M`) | `deu_news_2023_1M-sentences.txt` | Aktuelle Begriffe, Komposita, formelleres Deutsch | ~5 % | **CC BY 4.0** (Universität Leipzig / Wortschatz) | [`deu_news_2023_1M.tar.gz`](https://downloads.wortschatz-leipzig.de/corpora/deu_news_2023_1M.tar.gz) |
| **Tatoeba (Deutsch)** | `tatoeba_deu_sentences.tsv.bz2` | Sehr saubere, kurze Alltagssätze | ~3 % | **CC BY 2.0 FR** (einzelne Sätze teils CC0) | [`deu_sentences.tsv.bz2`](https://downloads.tatoeba.org/exports/per_language/deu/deu_sentences.tsv.bz2) |

---

## 2. Download & Einrichtung in `DataSet/`

### Automatischer Download (Tatoeba, OpenSubtitles, Leipzig News, FineWeb-2)
Für alle frei per HTTP/HuggingFace-Streaming beziehbaren Quellen gibt es ein Hilfsskript:

```bash
# Prüfen, welche Quellen bereits lokal vorhanden sind (kein Download)
python scripts/download_datasets.py --list

# Fehlende v1.5-Quellen herunterladen (FineWeb-2 standardmäßig auf ~600 MB Text begrenzt)
python scripts/download_datasets.py --sources tatoeba opensubtitles leipzig_news fineweb2
```

Optionale Parameter für `FineWeb-2`:
- `--fineweb-max-mb 600` steuert die unkomprimierte Ziel-Textmenge aus `deu_Latn`.
- `--fineweb-min-chars 120` verwirft extrem kurze Snippets bereits beim Streamen.

### Manuelle Quellen
1. **Reddit `r/FragReddit` (optional für Trainingskorpus):**
   Über das [Arctic Shift Download Utility](https://arctic-shift.photon-reddit.com/download-utility) Subreddit `FragReddit` (Posts & Comments als `.jsonl`) herunterladen und als `DataSet/r_fragreddit_posts.jsonl` sowie `DataSet/r_fragreddit_comments.jsonl` ablegen.
2. **Leipzig Corpora (Wikipedia 2021 & Web 2021):**
   Auf [wortschatz.uni-leipzig.de](https://wortschatz.uni-leipzig.de/en/download/German) die `1M`-Archive (`deu_wikipedia_2021_1M` und `deu-de_web_2021_1M`) laden, entpacken und die `*-sentences.txt` nach `DataSet/` kopieren.

> **Hinweis:** Fehlende Quellen werden von `scripts/build_corpus.py` automatisch mit einer Meldung übersprungen — der Korpus lässt sich auch mit einer Teilmenge der Quellen bauen.

---

## 3. Bereinigung, Filter & Held-out-Trennung

`scripts/build_corpus.py` wendet für alle Quellen einheitliche Qualitätsfilter sowie domänenspezifische Bereinigungen an:

1. **Allgemeine Textbereinigung (`clean_fragment`):**
   - Doppelte HTML-Entitätsdekodierung (`&amp;amp;` -> `&`) und Entfernung unsichtbarer Unicode-Zeichen (Zero-Width-Spaces, Soft-Hyphens).
   - Entfernung von Zitatzeilen (`>`), Markdown-Formatierung, URLs, Reddit-/Social-Handles (`u/`, `r/`, `@`, `#`) und HTML-Tags.
   - Längenfilter (`2` bis `80` Wörter), Buchstaben-/Leerzeichen-Quote (`>= 65 %`) sowie Heuristik gegen rein englische Kommentare.
   - **64-Bit-BLAKE2b-Deduplizierung** über alle Quellen hinweg.
2. **Spezialfilter `OpenSubtitles`:**
   - Entfernung von Regieanweisungen (`[...]`, `(...)`), Musiksymbolen (`♪`, `♫`), führenden Dialogstrichen (`- `) und Sprecher-Präfixen (`NAME: `).
   - Herausfiltern von Untertitel-Credits (`opensubtitles`, `sync by`, `untertitel`, `übersetzung`, `www.`).
   - **Dialog-Turn-Joining:** Verbindet bis zu 3 aufeinanderfolgende kurze Untertitelzeilen (bis max. 32 Wörter) zu natürlichen Gesprächs-Chunks.
3. **Proportionale Mischung & Cooldown-Korpus:**
   - Alle vorhandenen Quellen werden gemäß `config.CORPUS_SOURCE_WEIGHTS` ohne Oversampling ausbalanciert (kleine Quellen wie Tatoeba fließen komplett ein).
   - Zusätzlich wird `DataSet/corpus_de_v1_5_cooldown.txt` (kurze Alltagssätze aus `tatoeba`, `opensubtitles` und `fragreddit`, max. 300.000 Zeilen) für die Cooldown-Phase am Ende des Trainings abgespalten.
4. **Strikte Trennung zum Held-out-Benchmark (`DataSet/heldout_benchmark.jsonl`):**
   - Vor dem Schreiben des Trainingskorpus zieht `build_corpus.py` deterministisch (Seed `1337`) bis zu 300 Sätze pro Quelle (4–35 Wörter) als Held-out-Testset ab — oder liest ein bereits existierendes `heldout_benchmark.jsonl` ein und blockiert alle darin enthaltenen Hashes vorab.
   - Dadurch ist **0 % Datenleck** zwischen Training und Held-out-/KSR-Benchmark garantiert.

---

## 4. Korpus-Statistik-Protokoll

| Version | Quellen | Dateigröße | Zeilen (Chunks) | Wörter       | Gespeichert in |
| :--- | :--- | :--- |:----------------|:-------------| :--- |
| **v1** | ~85 % `r/FragReddit`, ~15 % `Wikipedia 2021` | `528 MB` | `4.363.852`     | `85.321.580` | `DataSet/corpus_de_v2.txt` |
| **v1.5** | FragReddit + OpenSubtitles + FineWeb-2 DE + Leipzig (Web/Wiki/News) + Tatoeba | `1,38 GB` (`1.377 MB`) | *`11,09 mio`*     | *`214,4 mio`*  | `DataSet/corpus_de_v1_5.txt` |

Nach jedem Lauf von `python scripts/build_corpus.py` wird automatisch `DataSet/corpus_de_v1_5_stats.json` mit exakten Zeilen-, Wörter-, Byte- und Prozentwerten pro Quelle aktualisiert.

---

## 5. Lizenz- & Attributions-Hinweise

- **OPUS OpenSubtitles:** Bereitgestellt über OPUS / OpenSubtitles.org. Bei Verwendung ist ein sichtbarer Link auf [http://www.opensubtitles.org/](http://www.opensubtitles.org/) sowie folgende Zitation erforderlich:
  > P. Lison and J. Tiedemann, 2016. *OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles.* Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016).
- **FineWeb-2 (`deu_Latn`):** Lizenziert unter **ODC-By v1.0** (HuggingFaceFW / Common Crawl). Namensnennung erforderlich.
- **Leipzig Corpora Collection:** Lizenziert unter **CC BY 4.0** (Universität Leipzig, Wortschatz).
- **Tatoeba (`deu`):** Lizenziert unter **CC BY 2.0 FR** (Tatoeba-Mitwirkende; einzelne Einträge CC0).
- **Reddit `r/FragReddit`:** Nutzerinhalte bleiben Eigentum der jeweiligen Autoren und unterliegen den Reddit-Nutzungsbedingungen; Rohdaten dürfen nicht öffentlich weiterverbreitet werden.
