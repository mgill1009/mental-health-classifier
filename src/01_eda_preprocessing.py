"""
01_eda_preprocessing.py
Loads, cleans, deduplicates, and splits the mental health dataset.
Outputs: data/processed/train.csv, val.csv, test.csv, class_weights.npy, label_map.json
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Paths 
RAW_PRIMARY   = Path("data/raw/primary/CombineData.csv")
RAW_TEST      = Path("data/raw/primary/mental_health_combined_test.csv")
RAW_SEC_1     = Path("data/raw/secondary/mental_heath_unbanlanced.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Label map 
LABEL_MAP = {
    "Normal": 0, "Stress": 1, "Anxiety": 1, "Personality disorder": 1,
    "Depression": 2, "Suicidal": 2, "Bipolar": 2,
}

# Text cleaning 
def fix_encoding(text):
    if not isinstance(text, str):
        return ""
    return text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")

def split_camel(text):
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

def clean_bert(text):
    text = fix_encoding(text)
    text = split_camel(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

NEGATIONS = {"not", "never", "no", "nor", "neither", "n't", "cant", "cannot",
             "dont", "wont", "wouldnt", "shouldnt", "couldnt", "didnt", "isnt",
             "arent", "wasnt", "werent", "havent", "hasnt", "hadnt"}

def clean_tfidf(text):
    text = clean_bert(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    # Remove stopwords but keep negations
    from nltk.corpus import stopwords
    stops = set(stopwords.words("english")) - NEGATIONS
    tokens = [t for t in tokens if t not in stops]
    return " ".join(tokens)

# Load & merge 
import nltk
nltk.download("stopwords", quiet=True)

primary = pd.read_csv(RAW_PRIMARY, usecols=["statement", "status"])
primary.columns = ["text", "label"]

secondary = pd.read_csv(RAW_SEC_1, usecols=["text", "label"], on_bad_lines="skip")

df = pd.concat([primary, secondary], ignore_index=True)
df.dropna(subset=["text", "label"], inplace=True)
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() >= 10].copy()

# Deduplicate 
df.drop_duplicates(subset="text", inplace=True)

# Map labels 
df = df[df["label"].isin(LABEL_MAP)].copy()
df["label"] = df["label"].map(LABEL_MAP)

# Clean text
df["text_bert"]  = df["text"].apply(clean_bert)
df["text_tfidf"] = df["text"].apply(clean_tfidf)
df = df[df["text_bert"].str.len() >= 5].copy()

# Load test set
test = pd.read_csv(RAW_TEST)
test.columns = ["text", "label"]
test = test[test["label"].isin(LABEL_MAP)].copy()
test["label"] = test["label"].map(LABEL_MAP)
test["text_bert"]  = test["text"].apply(clean_bert)
test["text_tfidf"] = test["text"].apply(clean_tfidf)

# Remove test texts from training pool
test_texts = set(test["text"].str.strip().str.lower())
df = df[~df["text"].str.strip().str.lower().isin(test_texts)].copy()

# Train / val split 
train, val = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])

# Class weights
classes = np.array([0, 1, 2])
weights = compute_class_weight("balanced", classes=classes, y=train["label"].values)
np.save(PROCESSED_DIR / "class_weights.npy", weights)

# Save
train.reset_index(drop=True).to_csv(PROCESSED_DIR / "train.csv", index=False)
val.reset_index(drop=True).to_csv(PROCESSED_DIR / "val.csv",   index=False)
test.reset_index(drop=True).to_csv(PROCESSED_DIR / "test.csv",  index=False)

with open(PROCESSED_DIR / "label_map.json", "w") as f:
    json.dump(LABEL_MAP, f, indent=2)

print(f"Done. Train={len(train)}, Val={len(val)}, Test={len(test)} | Weights={weights.round(4)}")
