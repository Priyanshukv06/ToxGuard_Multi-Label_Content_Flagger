"""
Production fine-tuning script for the DistilBERT multi-label toxicity classifier.

This is the script-ified, full-data version of notebooks/distilbert_transformer.ipynb
(same architecture, same conventions as the BiLSTM training scripts). Run this to
produce the artifacts the FastAPI backend actually loads.

Run in the GPU training env:

    conda activate toxguard-transformer
    python scripts/train_transformer.py

Outputs:
    models/transformer_model/            (HF backbone + tokenizer, via save_pretrained)
    models/transformer_model/classifier_head.pt   (the 768->6 linear head's state_dict)
    models/transformer_inference_artifacts.json    (per-class optimal thresholds, target_cols)
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

# ---- Config -------------------------------------------------------------
SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128          # cap on sequence length; actual padding is dynamic (per-batch), see batch_tokenize
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
TEXT_COL = "comment_text"
TARGET_COLS = ['toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']
NUM_CLASSES = len(TARGET_COLS)

# This is a portfolio project: the point of the transformer is to demonstrate the
# architecture/explainability comparison against the BiLSTM models, not to squeeze
# out maximum accuracy. DistilBERT is pretrained, so 150k labeled examples is a
# completely standard fine-tuning size. Training on the full ~2.9M rows would take
# much longer for a marginal (if any) accuracy gain — set to None only if you've
# confirmed 150k underperforms and specifically want to scale up.
SUBSAMPLE_TRAIN = 150_000

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "input")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_MODEL_DIR = os.path.join(MODELS_DIR, "transformer_model")
OUT_ARTIFACTS = os.path.join(MODELS_DIR, "transformer_inference_artifacts.json")

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretokenize(texts, tokenizer, max_len):
    """
    Tokenizes to token-ID LISTS (no padding yet) upfront, instead of one string
    at a time inside Dataset.__getitem__. Padding is applied per-batch in the
    DataLoader's collate_fn (dynamic padding) instead of forcing every sample to
    MAX_LEN — most comments are far shorter than 128 tokens, so fixed max-length
    padding wastes a large fraction of every forward/backward pass on padding
    tokens. Dynamic padding pads each batch only to its own longest sequence.
    """
    texts = [str(t) for t in texts]
    enc = tokenizer(texts, truncation=True, max_length=max_len)
    return enc["input_ids"]


class ToxicityDataset(Dataset):
    """Wraps PRE-TOKENIZED (unpadded) id lists — __getitem__ is pure indexing,
    no tokenizer calls. Padding happens later, per-batch, in collate_fn."""

    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}


def make_collate_fn(tokenizer):
    """Dynamically pads each batch to its own longest sequence (not a fixed MAX_LEN)."""
    def collate(batch):
        ids = [item["input_ids"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])
        padded = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        return {"input_ids": padded["input_ids"], "attention_mask": padded["attention_mask"], "labels": labels}
    return collate


class DistilBertMultiLabel(nn.Module):
    """DistilBERT backbone -> [CLS] hidden state -> dropout -> linear(768, num_classes)."""

    def __init__(self, model_name, num_classes, dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, output_attentions=False):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        cls_hidden = out.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_hidden))
        if output_attentions:
            return logits, out.attentions
        return logits


def run_epoch(model, loader, criterion, optimizer=None, scheduler=None, log_every=200):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    n_batches = len(loader)
    t0 = time.time()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with torch.set_grad_enabled(train):
            logits = model(input_ids, attn)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        total_loss += loss.item() * len(labels)
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        if train and (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            eta = (n_batches - step - 1) / rate
            print(f"    batch {step + 1}/{n_batches}  loss={loss.item():.4f}  "
                  f"{rate:.1f} batch/s  ETA this epoch: {eta / 60:.1f} min")

    avg_loss = total_loss / len(loader.dataset)
    probs = np.concatenate(all_probs)
    y = np.concatenate(all_labels)
    aucs = [
        roc_auc_score(y[:, j], probs[:, j]) if len(np.unique(y[:, j])) > 1 else float("nan")
        for j in range(NUM_CLASSES)
    ]
    return avg_loss, float(np.nanmean(aucs)), probs, y


def main():
    print(f"Device: {DEVICE}" + (f" - {torch.cuda.get_device_name(0)}" if DEVICE.type == "cuda" else " (CPU — this will be slow)"))

    print("\nLoading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_split.csv")).fillna({TEXT_COL: "missing_text"})
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_split.csv")).fillna({TEXT_COL: "missing_text"})
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_split.csv")).fillna({TEXT_COL: "missing_text"})

    if SUBSAMPLE_TRAIN is not None and len(train_df) > SUBSAMPLE_TRAIN:
        train_df = train_df.sample(n=SUBSAMPLE_TRAIN, random_state=SEED).reset_index(drop=True)

    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate_fn = make_collate_fn(tokenizer)

    print("Tokenizing train/val/test splits upfront (one-time cost, then training is GPU-bound)...")
    t_tok = time.time()
    train_ids = pretokenize(train_df[TEXT_COL].tolist(), tokenizer, MAX_LEN)
    val_ids = pretokenize(val_df[TEXT_COL].tolist(), tokenizer, MAX_LEN)
    test_ids = pretokenize(test_df[TEXT_COL].tolist(), tokenizer, MAX_LEN)
    print(f"  done in {time.time() - t_tok:.0f}s")

    train_ds = ToxicityDataset(train_ids, train_df[TARGET_COLS].values)
    val_ds = ToxicityDataset(val_ids, val_df[TARGET_COLS].values)
    test_ds = ToxicityDataset(test_ids, test_df[TARGET_COLS].values)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    print("\nBuilding model...")
    model = DistilBertMultiLabel(MODEL_NAME, NUM_CLASSES).to(DEVICE)
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    pos_counts = train_df[TARGET_COLS].sum().values
    neg_counts = len(train_df) - pos_counts
    pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float).to(DEVICE)
    print(f"  pos_weight: {dict(zip(TARGET_COLS, pos_weight.cpu().numpy().round(2)))}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    print(f"\nTraining for {EPOCHS} epochs ({total_steps} total steps)...")
    val_probs, val_y = None, None
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_auc, _, _ = run_epoch(model, train_loader, criterion, optimizer, scheduler)
        val_loss, val_auc, val_probs, val_y = run_epoch(model, val_loader, criterion)
        dt = time.time() - t0
        print(f"  epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
              f"  val_mean_auc={val_auc:.4f}  ({dt:.0f}s)")

    print("\n--- Per-class optimal thresholds (validation set) ---")
    optimal_thresholds = {}
    for j, c in enumerate(TARGET_COLS):
        precision, recall, thresh = precision_recall_curve(val_y[:, j], val_probs[:, j])
        f1 = np.divide(2 * precision * recall, precision + recall,
                        out=np.zeros_like(precision), where=(precision + recall) != 0)
        best_idx = int(np.argmax(f1[:-1]))
        optimal_thresholds[c] = float(thresh[best_idx])
        print(f"  {c:<18} optimal_threshold={thresh[best_idx]:.4f}  f1={f1[best_idx]:.4f}")

    print("\n--- Held-out test evaluation ---")
    _, test_auc, test_probs, test_y = run_epoch(model, test_loader, criterion)
    for j, c in enumerate(TARGET_COLS):
        auc = roc_auc_score(test_y[:, j], test_probs[:, j])
        ap = average_precision_score(test_y[:, j], test_probs[:, j])
        print(f"  {c:<18} ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}")

    print("\n--- Saving model & artifacts ---")
    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    model.backbone.save_pretrained(OUT_MODEL_DIR)
    tokenizer.save_pretrained(OUT_MODEL_DIR)
    torch.save(model.classifier.state_dict(), os.path.join(OUT_MODEL_DIR, "classifier_head.pt"))

    inference_artifacts = {
        "optimal_thresholds": optimal_thresholds,
        "target_cols": TARGET_COLS,
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
    }
    with open(OUT_ARTIFACTS, "w") as f:
        json.dump(inference_artifacts, f, indent=4)

    print(f"✅ Saved backbone+tokenizer to {OUT_MODEL_DIR}")
    print(f"✅ Saved classifier head to {os.path.join(OUT_MODEL_DIR, 'classifier_head.pt')}")
    print(f"✅ Saved thresholds to {OUT_ARTIFACTS}")


if __name__ == "__main__":
    main()
