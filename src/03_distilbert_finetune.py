"""
03_distilbert_finetune.py
Fine-tunes distilbert-base-uncased on the processed training data.
Outputs: models/distilbert/distilbert-mental-health/ (HuggingFace model dir)

Run on a GPU machine or Google Colab (T4 recommended).
Install: pip install torch transformers accelerate scikit-learn pandas numpy
"""

import numpy as np
import pandas as pd
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, classification_report

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_LENGTH   = 256
BATCH_SIZE   = 32
LR           = 2e-5
NUM_EPOCHS   = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
PATIENCE     = 2
NUM_LABELS   = 3
HF_REPO      = None   # set to "your-hf-username/model-name" to push to HuggingFace Hub

PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("models/distilbert/distilbert-mental-health")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────
train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
val_df   = pd.read_csv(PROCESSED_DIR / "val.csv")
test_df  = pd.read_csv(PROCESSED_DIR / "test.csv")
weights  = np.load(PROCESSED_DIR / "class_weights.npy")

# ── Tokeniser ─────────────────────────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

# ── Dataset ───────────────────────────────────────────────────────────────
class MentalHealthDataset(Dataset):
    def __init__(self, df):
        self.texts  = df["text_bert"].fillna("").tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_ds = MentalHealthDataset(train_df)
val_ds   = MentalHealthDataset(val_df)
test_ds  = MentalHealthDataset(test_df)

# ── WeightedRandomSampler (batch-level class balancing) ───────────────────
sample_weights = torch.tensor([weights[l] for l in train_df["label"].tolist()])
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# ── Model ─────────────────────────────────────────────────────────────────
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)

# ── Custom trainer with weighted loss ─────────────────────────────────────
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(model.device)
        )
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

# ── Metrics ───────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

# ── Training args ─────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=str(MODEL_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    report_to="none",
)

# ── Train ─────────────────────────────────────────────────────────────────
trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
)

trainer.train()

# ── Evaluate on test set ──────────────────────────────────────────────────
preds_out = trainer.predict(test_ds)
preds     = np.argmax(preds_out.predictions, axis=-1)
labels    = test_df["label"].values
macro_f1  = f1_score(labels, preds, average="macro")

# ── Save model ────────────────────────────────────────────────────────────
trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

if HF_REPO:
    model.push_to_hub(HF_REPO)
    tokenizer.push_to_hub(HF_REPO)

print(
    f"Done. "
    f"Test macro-F1={macro_f1:.4f} | "
    f"High Risk recall={classification_report(labels, preds, output_dict=True)['2']['recall']:.4f} | "
    f"Model saved to {MODEL_DIR}"
)
