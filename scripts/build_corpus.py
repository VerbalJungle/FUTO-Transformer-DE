import os
import re
import random
import xml.etree.ElementTree as ET
from datasets import load_dataset

OUTPUT_FILE = "corpus_de_expert.txt"
all_sentences = set()  # Set verhindert Duplikate automatisch

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
    # URLs, Twitter-Handles und Hashtags entfernen
    text = re.sub(r'https?://\S+|www\.\S+|@\w+|#\w+', '', text)
    # HTML/XML Tags entfernen
    text = re.sub(r'<.*?>', '', text)
    # Satzzeichen-Ketten zähmen
    text = re.sub(r'([!?.,])\1+', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    if 3 <= len(words) <= 30:
        return text
    return None

def inject_qwertz_typos(text):
    chars = list(text)
    new_chars = []
    i = 0
    while i < len(chars):
        c = chars[i]
        c_lower = c.lower()
        # Doppelbuchstaben auslassen: "kommen" -> "komen"
        if i + 1 < len(chars) and chars[i] == chars[i+1] and random.random() < 0.35:
            new_chars.append(c)
            i += 2
            continue
        # Nachbartaste treffen
        if c_lower in QWERTZ_MAP and random.random() < 0.08:
            neighbor = random.choice(QWERTZ_MAP[c_lower])
            new_chars.append(neighbor.upper() if c.isupper() else neighbor)
        else:
            new_chars.append(c)
        i += 1
    return "".join(new_chars)

# 1. Leipzig Corpora einlesen (tab-getrennt: id \t sentence)
leipzig_files = [
    "deu-de_web_2021_1M-sentences.txt",     #[cite: 1]
    "deu_wikipedia_2021_1M-sentences.txt",  #[cite: 4]
    "deu_mixed-typical_2011_1M-sentences.txt", #[cite: 2]
    "deu_wikipedia_2016_1M-sentences.txt"   #[cite: 3]
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
chat_files = [f for f in os.listdir(".") if f.startswith("chatkorpus") and f.endswith(".xml")]
for cf in chat_files:
    count = 0
    try:
        # Schneller Regex-Scan auf <messageBody>...</messageBody>[cite: 5]
        with open(cf, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        bodies = re.findall(r'<messageBody>(.*?)</messageBody>', content, re.DOTALL) #[cite: 5]
        for b in bodies:
            s = clean_sentence(b)
            if s:
                all_sentences.add(s)
                count += 1
        print(f"  -> {count:,} Chat-Nachrichten aus '{cf}' geladen.")
    except Exception as e:
        print(f"Fehler bei Chatkorpus: {e}")

# 3. OpenAssistant (OASST-DE)
print("\n=== 3. Lade OpenAssistant (OASST-DE) via HuggingFace ===")
try:
    oasst = load_dataset("OpenAssistant/OASST-DE", split="train")
    count = 0
    for item in oasst:
        for raw_s in re.split(r'(?<=[.!?])\s+', item.get("text", "")):
            s = clean_sentence(raw_s)
            if s:
                all_sentences.add(s)
                count += 1
    print(f"  -> {count:,} Sätze aus OASST-DE geladen.")
except Exception as e:
    print(f"  OASST-DE Fehler: {e}")

# 4. German Hate Speech Superset (Social Media & Web-Kommentare)
print("\n=== 4. Lade Social-Media-Korpus (Superset) ===")
try:
    superset = load_dataset("manueltonneau/german-hate-speech-superset", split="train")
    count = 0
    for item in superset:
        # Extrahiere reinen Text
        text = item.get("text", "")
        for raw_s in re.split(r'(?<=[.!?\n])\s+', text):
            s = clean_sentence(raw_s)
            if s:
                all_sentences.add(s)
                count += 1
                if count >= 800_000:  # Max 800k für saubere Balance
                    break
    print(f"  -> {count:,} Sätze aus Social Media Superset geladen.")
except Exception as e:
    print(f"  Superset Fehler: {e}")

# 5. Tippfehler-Augmentierung & Finale Mischung
print("\n=== 5. Generiere finale Mischung & QWERTZ-Tippfehler ===")
final_list = list(all_sentences)
augmented_list = []

for s in final_list:
    augmented_list.append(s)
    # 20% Tippfehler-Varianten einmischen (lernt "komen" -> "kommen")
    if random.random() < 0.20:
        augmented_list.append(inject_qwertz_typos(s))

random.seed(42)
random.shuffle(augmented_list)

print(f"Schreibe {len(augmented_list):,} Sätze nach '{OUTPUT_FILE}'...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for s in augmented_list:
        f.write(s + "\n")

print("\nDatensatz-Erstellung erfolgreich abgeschlossen!")
