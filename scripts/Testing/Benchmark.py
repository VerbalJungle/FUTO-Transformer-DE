import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from llama_cpp import Llama

MODEL_PATH = "../../Models/GermanV1-0_9_1-unquantized.gguf"

print(f"Lade '{MODEL_PATH}' für den Stresstest...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=128,
    n_threads=4,
    verbose=False
)

# Test-Szenarien mit erwarteten Zielwörtern / Wortanfängen
TEST_SUITE = {
    "1. Entitäten & Geografie (Faktenwissen)": [
        ("Costa ", ["Rica"]),
        ("San ", ["Francisco", "Diego"]),
        ("New ", ["York"]),
        ("Deutsche ", ["Bahn", "Telekom", "Post", "Bank"]),
        ("Nordrhein-", ["Westfalen"]),
        ("Buenos ", ["Aires"]),
    ],
    "2. Alltagsphrasen & Chat-Dynamik": [
        ("Guten ", ["Morgen", "Tag", "Abend"]),
        ("Vielen Dank für ", ["die", "deine", "das", "Ihre"]),
        ("Ich melde mich ", ["gleich", "später", "nachher", "morgen"]),
        ("Bin gerade ", ["unterwegs", "noch", "im", "beim"]),
        ("Wann bist du ", ["da", "zu", "fertig"]),
        ("Schönes ", ["Wochenende", "Fest"]),
    ],
    "3. Grammatik & Nebensatz-Verbendstellung": [
        ("Ich bleibe heute daheim, weil ich krank ", ["bin", "geworden"]),
        ("Er fährt morgen mit dem ", ["Zug", "Auto", "Bus", "Fahrrad"]),
        ("Trotz des schlechten ", ["Wetters"]),
        ("Wir haben uns gestern lange ", ["unterhalten", "gesehen", "getroffen"]),
    ],
    "4. Wortvervollständigung (Teilwörter)": [
        ("Ich fahre zum Hauptbahn", ["hof"]),
        ("Hast du meine Nachr", ["icht"]),
        ("Wir müssen noch eink", ["aufen"]),
        ("Können wir kurz telefon", ["ieren"]),
    ],
    "5. Tippfehler-Kontext & Autokorrektur": [
        ("Wollen wir noch was essen gehn oder kom", ["mst", "men"]),
        ("Ich kann heute leider nit", ["kann", "nicht"]),
        ("Bis späta", ["ter", "r"]),
    ]
}

print("\n" + "=" * 65)
print("              FUTO KEYBOARD BENCHMARK & STRESSTEST")
print("=" * 65)

total_tests = 0
passed_top1 = 0
passed_top3 = 0
latencies = []

for category, test_cases in TEST_SUITE.items():
    print(f"\n** {category} **")
    for prompt, expected_words in test_cases:
        total_tests += 1
        
        # Latenz für die Vorhersage messen
        t0 = time.perf_counter()
        output = llm(
            prompt,
            max_tokens=2,
            stop=["\n", " "],
            temperature=0.0,  # Deterministisch: Greife die wahrscheinlichsten Tokens ab
            echo=False
        )
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        
        predicted = output["choices"][0]["text"].strip()
        
        # Prüfen, ob eine der erwarteten Optionen getroffen wurde
        is_hit = any(predicted.lower().startswith(exp.lower()) or exp.lower().startswith(predicted.lower()) 
                     for exp in expected_words if predicted)
        
        if is_hit:
            passed_top1 += 1
            passed_top3 += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            
        print(f"  {status:<6} | '{prompt}' -> '{predicted}' | Erwartet: {expected_words} ({latency:.1f} ms)")

# Auswertung
avg_latency = sum(latencies) / len(latencies)
accuracy = (passed_top1 / total_tests) * 100

print("\n" + "=" * 65)
print("                    BENCHMARK ERGEBNIS")
print("=" * 65)
print(f"Gesamtanzahl Tests:       {total_tests}")
print(f"Erfolgreich vorhergesagt: {passed_top1} / {total_tests} ({accuracy:.1f} %)")
print(f"Durchschnittliche Latenz: {avg_latency:.2f} ms pro Vorhersage")
print("=" * 65)
