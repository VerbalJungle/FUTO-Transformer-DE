import os
import io
import json
import re
import random
import xml.etree.ElementTree as ET
from datasets import load_dataset

OUTPUT_FILE = "corpus_de_expert1.txt"
all_sentences = set()  # verhindert Duplikate

# QWERTZ-Tastatur-Nachbartasten
QWERTZ_MAP = {
    'a': ['q', 'w', 's', 'y'], 'b': ['v', 'g', 'h', 'n'], 'c': ['x', 'd', 'f', 'v'],
    'd': ['s', 'e', 'r', 'f', 'x', 'c'], 'e': ['w', 's', 'd', 'r', '3'],
    'f': ['d', 'r', 't', 'g', 'c', 'v'], 'g': ['f', 't', 'z', 'h', 'v', 'b'],
    'h': ['g', 'z', 'u', 'j', 'b', 'n'], 'i': ['u', 'j', 'k', 'o', '8', '9'],
    'j': ['h', 'u', 'i', 'k', 'n', 'm'], 'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p', 'ö'], 'm': ['n', 'j', 'k'], 'n': ['b', 'h', 'j', 'm'],
    'o': ['i', 'k', 'l', 'p', '9', '0'], 'p': ['o', 'l', 'ö', 'ü'],
    'q': ['1', '2', 'w', 'a'], 'r': ['e', 'd', 'f', 't', '4', '5'],
    's': ['a', 'w', 'e', 'd', 'y', 'x'], 't': ['r', 'f', 'g', 'z', '5', '6'],
    'u': ['z', 'h', 'j', 'i', '7', '8'], 'v': ['c', 'f', 'g', 'b'],
    'w': ['q', 'a', 's', 'e', '2', '3'], 'x': ['y', 's', 'd', 'c'],
    'y': ['a', 's', 'x'], 'z': ['t', 'g', 'h', 'u', '6', '7'],
    'ä': ['ö', 'p', 'ü'], 'ö': ['l', 'p', 'ä', 'ü'], 'ü': ['p', 'ö', 'ä']
}


def clean_sentence(text):
    if not text or not isinstance(text, str):
        return None
    # URLs, Twitter/Reddit-Handles und Hashtags entfernen
    text = re.sub(r'https?://\S+|www\.\S+|@\w+|#\w+|/?u/\w+|/?r/\w+', '', text)
    # Markdown-Links [Text](URL) -> Text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Markdown-Formatierungen (Fett, Kursiv, Spoiler, Strikethrough)
    text = re.sub(r'[*_~`>!]', '', text)
    # HTML/XML Tags entfernen
    text = re.sub(r'<.*?>', '', text)
    # Satzzeichen-Ketten zähmen
    text = re.sub(r'([!?.,])\1+', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    # 3-25 Wörter
    if 3 <= len(words) <= 25:
        return text
    return None


def iter_reddit_lines(filepath):
    """Liest Zeilen aus .jsonl Dateien."""
    if filepath.endswith('.jsonl'):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                yield line
    else:
        print(f"Datei {filepath} ist kein jsonl")


def inject_qwertz_typos(text):
    chars = list(text)
    new_chars = []
    i = 0
    while i < len(chars):
        c = chars[i]
        c_lower = c.lower()
        # Doppelbuchstaben auslassen
        if i + 1 < len(chars) and chars[i] == chars[i + 1] and random.random() < 0.35:
            new_chars.append(c)
            i += 2
            continue
        # Nachbartasten
        if c_lower in QWERTZ_MAP and random.random() < 0.08:
            neighbor = random.choice(QWERTZ_MAP[c_lower])
            new_chars.append(neighbor.upper() if c.isupper() else neighbor)
        else:
            new_chars.append(c)
        i += 1
    return "".join(new_chars)


# 1. Leipzig Corpora einlesen (tab-getrennt: id \t sentence)
leipzig_files = [
    "../DataSet/deu-de_web_2021_1M-sentences.txt",  # [cite: 1]
    "../DataSet/deu_wikipedia_2021_1M-sentences.txt",  # [cite: 4]
    "../DataSet/deu_mixed-typical_2011_1M-sentences.txt",  # [cite: 2]
    # "../DataSet/deu_wikipedia_2016_1M-sentences.txt"  # [cite: 3]
]

print("=== 1. Lade Leipzig Corpora ===")
for lf in leipzig_files:
    if os.path.exists(lf):
        count = 0
        with open(lf, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                raw = parts[1] if len(parts) > 1 else parts[0]
                s = clean_sentence(raw)
                if s:
                    all_sentences.add(s)
                    count += 1
        print(f"  -> {count:,} Sätze aus '{lf}' geladen.")
    else:
        print(f"  (Datei '{lf}' nicht gefunden, wird übersprungen)")

# 2. Dortmunder Chatkorpus (XML / messageBody)
print("\n=== 2. Lade Chatkorpus ===")
chat_files = [f for f in os.listdir("../DataSet/") if f.startswith("chatkorpus") and f.endswith(".xml")]
for cf in chat_files:
    count = 0
    try:
        # Schneller Regex-Scan auf <messageBody>...</messageBody>[cite: 5]
        with open(cf, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        bodies = re.findall(r'<messageBody>(.*?)</messageBody>', content, re.DOTALL)  # [cite: 5]
        for b in bodies:
            s = clean_sentence(b)
            if s:
                all_sentences.add(s)
                count += 1
        print(f"  -> {count:,} Chat-Nachrichten aus '{cf}' geladen.")
    except Exception as e:
        print(f"Fehler bei Chatkorpus: {e}")

# 3. OpenAssistant (OASST-DE)
# print("\n=== 3. Lade OpenAssistant (OASST-DE) via HuggingFace ===")
# try:
# oasst = load_dataset("OpenAssistant/OASST-DE", split="train")
# count = 0
# for item in oasst:
# for raw_s in re.split(r'(?<=[.!?])\s+', item.get("text", "")):
#   s = clean_sentence(raw_s)
#  if s:
#     all_sentences.add(s)
#    count += 1
# print(f"  -> {count:,} Sätze aus OASST-DE geladen.")
# except Exception as e:
#   print(f"  OASST-DE Fehler: {e}")

# 4. FragReddit Posts (160k Posts und ~3mio Kommentare)
reddit_posts_files = ['../DataSet/r_fragreddit_comments.jsonl', '../DataSet/r_fragreddit_posts.jsonl']
print(f"\n=== 4. Lade FragReddit Posts aus '{reddit_posts_files}' ===")
count = 0
for file in reddit_posts_files:
    for line in iter_reddit_lines(file):
        try:
            data = json.loads(line)

            if data.get('author') == 'AutoModerator':
                continue

            title = data.get('title', '')
            selftext = data.get('selftext', '')
            body = data.get('body', '')

            # Nur Titel und Text von Posts bzw. Kommentaren
            for text_segment in [title, selftext, body]:
                if (
                        not text_segment
                        or text_segment in ['[deleted]', '[removed]']
                        or 'Automoderator' in text_segment
                ):
                    continue

                for raw_s in re.split(r'(?<=[.!?\n])\s+', text_segment):
                    s = clean_sentence(raw_s)
                    if s:
                        all_sentences.add(s)
                        count += 1

        except json.JSONDecodeError:
            # nur Json Fehler
            continue
print(f'  -> {count:,} Sätze aus FragReddit Posts extrahiert.')

# 5 Finale Mischung für Next-Word Vorhersage
print("\n=== 5. Generiere finale Mischung ===")
augmented_list = list(all_sentences)
random.seed(42)
random.shuffle(augmented_list)

print(f"Schreibe {len(augmented_list):,} Sätze nach '{OUTPUT_FILE}'...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for s in augmented_list:
        f.write(s + "\n")

print("\nDatensatz-Erstellung erfolgreich abgeschlossen!")
