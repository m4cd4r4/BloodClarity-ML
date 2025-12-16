#!/usr/bin/env python3
"""
TinyBERT NER Training Script for Biomarker Extraction
Fine-tunes TinyBERT on synthetic lab report data for Named Entity Recognition
"""

import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from sklearn.metrics import classification_report, f1_score

# Configuration
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"  # 14.5M parameters
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 5e-5

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "ml" / "synthetic"
OUTPUT_DIR = SCRIPT_DIR.parent.parent / "models" / "tinybert-ner"

# Label mapping
LABEL_LIST = ["O", "B-BIOMARKER", "I-BIOMARKER", "B-VALUE", "I-VALUE", "B-UNIT", "I-UNIT", "B-RANGE", "I-RANGE", "B-STATUS", "I-STATUS"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


class BiomarkerNERDataset(Dataset):
    """Dataset for biomarker NER training"""

    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for sample in data['samples']:
            # Extract tokens and labels
            tokens = [t['token'] for t in sample['tokens']]
            labels = [t['label'] for t in sample['tokens']]

            # Convert labels to IDs
            label_ids = [LABEL2ID.get(l, LABEL2ID['O']) for l in labels]

            self.samples.append({
                'tokens': tokens,
                'labels': label_ids,
                'text': sample['text']
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample['tokens']
        labels = sample['labels']

        # Tokenize with word-level alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Align labels with tokenized output
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss)
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a word gets the label
                if word_idx < len(labels):
                    aligned_labels.append(labels[word_idx])
                else:
                    aligned_labels.append(-100)
            else:
                # Subsequent tokens of the same word
                # Use I- version of the label if it exists
                if word_idx < len(labels):
                    label = labels[word_idx]
                    label_name = ID2LABEL[label]
                    if label_name.startswith('B-'):
                        i_label = 'I-' + label_name[2:]
                        aligned_labels.append(LABEL2ID.get(i_label, label))
                    else:
                        aligned_labels.append(label)
                else:
                    aligned_labels.append(-100)
            previous_word_idx = word_idx

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels)
        }


def compute_metrics(eval_pred):
    """Compute F1 metrics for NER evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        true_preds = []
        true_labs = []
        for p, l in zip(prediction, label):
            if l != -100:
                true_preds.append(ID2LABEL[p])
                true_labs.append(ID2LABEL[l])
        true_predictions.append(true_preds)
        true_labels.append(true_labs)

    # Flatten for sklearn metrics
    flat_preds = [p for seq in true_predictions for p in seq]
    flat_labels = [l for seq in true_labels for l in seq]

    # Calculate F1 scores
    f1_micro = f1_score(flat_labels, flat_preds, average='micro', zero_division=0)
    f1_macro = f1_score(flat_labels, flat_preds, average='macro', zero_division=0)

    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
    }


def main():
    print("=" * 60)
    print("TinyBERT NER Training for Biomarker Extraction")
    print("=" * 60)
    print()

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load datasets
    print("Loading training data...")
    train_path = DATA_DIR / "train.json"
    test_path = DATA_DIR / "test.json"

    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        sys.exit(1)

    train_dataset = BiomarkerNERDataset(train_path, tokenizer, MAX_LENGTH)
    test_dataset = BiomarkerNERDataset(test_path, tokenizer, MAX_LENGTH)

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print()

    # Load model
    print("Loading TinyBERT model...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        report_to="none",  # Disable wandb/tensorboard
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train
    print("Starting training...")
    print("-" * 40)
    trainer.train()

    # Evaluate
    print()
    print("Final evaluation...")
    print("-" * 40)
    results = trainer.evaluate()
    print(f"F1 Micro: {results['eval_f1_micro']:.4f}")
    print(f"F1 Macro: {results['eval_f1_macro']:.4f}")

    # Save model
    print()
    print("Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save label mapping
    with open(OUTPUT_DIR / "label_map.json", 'w') as f:
        json.dump({'label2id': LABEL2ID, 'id2label': ID2LABEL}, f, indent=2)

    print(f"Model saved to: {OUTPUT_DIR}")

    # Test inference
    print()
    print("Testing inference...")
    print("-" * 40)

    test_texts = [
        "Hemoglobin: 14.5 g/dL (Normal)",
        "Glucose 126 mg/dL HIGH",
        "LDL Cholesterol: 145 mg/dL (Ref: 0-100)"
    ]

    model.eval()
    model.to(device)

    for text in test_texts:
        tokens = text.split()
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            return_tensors='pt'
        )

        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=2)

        word_ids = encoding['input_ids'][0].tolist()
        pred_labels = predictions[0].tolist()

        print(f"\nInput: {text}")
        print("Predictions:")

        word_idx_map = tokenizer(tokens, is_split_into_words=True).word_ids()
        seen_words = set()

        for i, (token_id, label_id) in enumerate(zip(word_ids, pred_labels)):
            if i < len(word_idx_map) and word_idx_map[i] is not None:
                word_idx = word_idx_map[i]
                if word_idx not in seen_words and word_idx < len(tokens):
                    label = ID2LABEL[label_id]
                    if label != 'O':
                        print(f"  {tokens[word_idx]}: {label}")
                    seen_words.add(word_idx)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
