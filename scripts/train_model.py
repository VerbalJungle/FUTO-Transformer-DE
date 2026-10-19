import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["ROCR_VISIBLE_DEVICES"] = "0"

DATASET_ID = "VerbalJungle/German-KeyboardLM-Corpus"  #HuggingFace Dataset Identifier
SPM_PATH = "tokenizer/futo_de_15k.model"
LOCAL_FALLBACK_FILE = "corpus_de_expert.txt"
MAX_LENGTH = 256

import torch
import sentencepiece as spm
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, load_dataset

# GPU-Prüfung: Muss zwingend mindestens 1 sein!
gpu_count = torch.cuda.device_count()
print(f"Sichtbare GPUs in PyTorch: {gpu_count}")
if gpu_count == 0:
    raise RuntimeError("FEHLER: ROCm erkennt keine GPU! Training wird nicht auf CPU gestartet.")

device_name = torch.cuda.get_device_name(0)
print(f"Aktives Gerät: {device_name}")

# 1. Tokenizer laden & Daten einlesen
SPM_PATH = "../tokenizer/futo_de_15k.model"
sp = spm.SentencePieceProcessor()
sp.load(SPM_PATH)

print("Lese und tokenisiere 'corpus_de_expert1.txt'...")
input_ids_list = []
with open("../DataSet/corpus_de_expert.txt", "r", encoding="utf-8") as f:
    for line in f:
        text = line.strip()
        if text:
            # 2 = BOS, 3 = EOSimport gguf
            ids = [2] + sp.encode_as_ids(text) + [3]
            # Begrenzung auf Kontextlänge
            if len(ids) <= 128:
                input_ids_list.append(ids)

print(f"{len(input_ids_list):,} Sätze im VRAM-Format vorbereitet.")
train_dataset = Dataset.from_dict({"input_ids": input_ids_list})

# 2. Modell-Architektur (FUTO 33M)
config = LlamaConfig(
    vocab_size=15008,
    hidden_size=512,
    intermediate_size=1376,
    num_hidden_layers=8,
    num_attention_heads=8,
    max_position_embeddings=2048,
    bos_token_id=2,
    eos_token_id=3,
    pad_token_id=0,
    tie_word_embeddings=True
)
model = LlamaForCausalLM(config)

# 3. Schneller Data-Collator mit Padding
def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, labels = [], []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [0] * pad_len)
        labels.append(ids + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

# 4. Trainingsparameter für RX 7900 XT
training_args = TrainingArguments(
    output_dir="./futo_de_checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=256,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="no",
    fp16=True,
    dataloader_num_workers=6,
    dataloader_pin_memory=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn
)

print(f"Starte GPU-Training auf {device_name}...")
trainer.train()

# Modell exportieren
trainer.save_model("./futo_de_final")
print("Training abgeschlossen! Modell in './futo_de_final' gespeichert.")