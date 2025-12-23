#!/usr/bin/env python3
"""
Example Training Pipeline for BloodClarity TinyBERT NER Model

This script demonstrates how to use the synthetic data for training.
Run this AFTER generating synthetic data with generate_synthetic_data.py

Requirements:
- transformers
- torch
- datasets
- accelerate
- seqeval (for NER metrics)

Install with: pip install transformers torch datasets accelerate seqeval
"""

import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


def load_synthetic_data(file_path: str) -> List[Dict]:
    """Load synthetic dataset from JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_dataset(data: List[Dict], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train/validation/test sets.

    Args:
        data: Full dataset
        train_ratio: Proportion for training (default: 0.8)
        val_ratio: Proportion for validation (default: 0.1)
        test_ratio: Proportion for testing (default: 0.1)

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    import random
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def convert_to_huggingface_format(data: List[Dict]) -> List[Dict]:
    """
    Convert synthetic data to Hugging Face NER format.

    Format expected by transformers NER:
    {
        "id": "sample_000000",
        "tokens": ["Vitamin", "D", ":", "45.2", "nmol/L"],
        "ner_tags": ["B-BIOMARKER_NAME", "I-BIOMARKER_NAME", "O", "B-BIOMARKER_VALUE", "B-BIOMARKER_UNIT"]
    }
    """
    hf_data = []

    for sample in data:
        hf_sample = {
            "id": sample["id"],
            "tokens": [token["text"] for token in sample["tokens"]],
            "ner_tags": [token["tag"] for token in sample["tokens"]]
        }
        hf_data.append(hf_sample)

    return hf_data


def create_label_mapping(data: List[Dict]) -> Dict:
    """
    Create label to ID mapping for NER tags.

    Returns:
        Dictionary with label2id and id2label mappings
    """
    labels = set()

    for sample in data:
        for token in sample["tokens"]:
            labels.add(token["tag"])

    labels = sorted(list(labels))  # Sort for consistency

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    return {
        "label2id": label2id,
        "id2label": id2label,
        "labels": labels
    }


def analyze_dataset(data: List[Dict]):
    """Print analysis of dataset."""
    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)

    total_samples = len(data)
    total_tokens = sum(len(sample["tokens"]) for sample in data)
    total_entities = sum(len(sample["entities"]) for sample in data)

    # Entity type distribution
    entity_types = defaultdict(int)
    for sample in data:
        for entity in sample["entities"]:
            entity_types[entity["label"]] += 1

    # Format distribution
    format_dist = defaultdict(int)
    for sample in data:
        format_dist[sample["format"]] += 1

    print(f"\nTotal Samples: {total_samples}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Average Tokens per Sample: {total_tokens / total_samples:.1f}")
    print(f"Total Entities: {total_entities}")
    print(f"Average Entities per Sample: {total_entities / total_samples:.1f}")

    print("\nEntity Type Distribution:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count} ({count/total_entities*100:.1f}%)")

    print("\nFormat Distribution:")
    for format_name, count in sorted(format_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {format_name}: {count} ({count/total_samples*100:.1f}%)")


def save_huggingface_dataset(data: List[Dict], output_dir: str, split_name: str):
    """Save data in Hugging Face format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hf_data = convert_to_huggingface_format(data)

    with open(output_path / f"{split_name}.json", 'w', encoding='utf-8') as f:
        json.dump(hf_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(hf_data)} samples to {output_path / f'{split_name}.json'}")


def example_training_setup():
    """
    Example code showing how to set up training with transformers.

    NOTE: This is pseudocode - requires transformers, torch, etc. to be installed.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE TRAINING SETUP (Pseudocode)")
    print("=" * 70)

    code = '''
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import load_dataset
import torch

# 1. Load the dataset
dataset = load_dataset('json', data_files={
    'train': 'data/ml/train.json',
    'validation': 'data/ml/val.json',
    'test': 'data/ml/test.json'
})

# 2. Load TinyBERT tokenizer and model
model_name = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define label mappings
label2id = {...}  # From create_label_mapping()
id2label = {...}

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# 3. Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=512
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)  # Subword tokens
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# 4. Set up training
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

# 5. Define metrics
from seqeval.metrics import classification_report, f1_score

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "f1": f1_score(true_labels, true_predictions),
    }

# 6. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 7. Train!
trainer.train()

# 8. Evaluate
results = trainer.evaluate(tokenized_datasets["test"])
print(results)

# 9. Save model
trainer.save_model("./bloodclarity-tinybert-ner")
tokenizer.save_pretrained("./bloodclarity-tinybert-ner")
'''

    print(code)


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare synthetic data for TinyBERT training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="C:/Scratch/bloodclarity/data/ml/synthetic_lab_reports.json",
        help="Input synthetic data JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="C:/Scratch/bloodclarity/data/ml/",
        help="Output directory for train/val/test splits"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BloodClarity TinyBERT Training Pipeline")
    print("=" * 70)

    # Load data
    print(f"\nLoading synthetic data from: {args.input}")
    data = load_synthetic_data(args.input)
    print(f"Loaded {len(data)} samples")

    # Analyze
    analyze_dataset(data)

    # Split
    print("\n" + "=" * 70)
    print("SPLITTING DATASET")
    print("=" * 70)
    train_data, val_data, test_data = split_dataset(
        data,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )
    print(f"Train: {len(train_data)} samples")
    print(f"Validation: {len(val_data)} samples")
    print(f"Test: {len(test_data)} samples")

    # Create label mapping
    print("\n" + "=" * 70)
    print("CREATING LABEL MAPPING")
    print("=" * 70)
    label_info = create_label_mapping(data)
    print(f"Found {len(label_info['labels'])} unique labels:")
    for label in label_info['labels']:
        print(f"  {label}")

    # Save label mapping
    label_path = Path(args.output_dir) / "label_mapping.json"
    with open(label_path, 'w') as f:
        json.dump(label_info, f, indent=2)
    print(f"\nSaved label mapping to: {label_path}")

    # Save splits
    print("\n" + "=" * 70)
    print("SAVING SPLITS")
    print("=" * 70)
    save_huggingface_dataset(train_data, args.output_dir, "train")
    save_huggingface_dataset(val_data, args.output_dir, "val")
    save_huggingface_dataset(test_data, args.output_dir, "test")

    # Show example training code
    example_training_setup()

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Install training dependencies: pip install transformers torch datasets accelerate seqeval")
    print("2. Adapt the example training code above to your needs")
    print("3. Run training with: python your_training_script.py")
    print("4. Evaluate on test set")
    print("5. Deploy model to BloodClarity app")


if __name__ == "__main__":
    main()
