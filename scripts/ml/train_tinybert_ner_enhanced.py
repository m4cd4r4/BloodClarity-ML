"""
Enhanced TinyBERT NER Training Script with Multi-Task Learning

Implements 3 joint tasks:
1. Named Entity Recognition (NER) - Biomarker, Value, Unit, Range extraction
2. Format Classification - Identify lab format from text patterns
3. Unit Prediction - Predict expected unit for biomarker context-awareness

Target: 98%+ accuracy on biomarker extraction

Requirements:
    pip install transformers torch datasets scikit-learn onnx onnxruntime seqeval

Usage:
    python train_tinybert_ner_enhanced.py --data_dir data/ml/synthetic --output_dir models/tinybert-ner-enhanced
"""

import json
import os
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, accuracy_score, f1_score
from seqeval.metrics import classification_report as ner_classification_report
from seqeval.metrics import f1_score as ner_f1_score
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NER label mapping (BIO tagging)
NER_LABELS = [
    'O',           # Outside any entity
    'B-BIOMARKER', # Beginning of biomarker name
    'I-BIOMARKER', # Inside biomarker name
    'B-VALUE',     # Beginning of measurement value
    'I-VALUE',     # Inside measurement value
    'B-UNIT',      # Beginning of unit
    'I-UNIT',      # Inside unit
    'B-RANGE',     # Beginning of reference range
    'I-RANGE',     # Inside reference range
    'B-FLAG',      # Beginning of abnormal flag (H/L)
    'I-FLAG',      # Inside abnormal flag
]

NER_LABEL_TO_ID = {label: idx for idx, label in enumerate(NER_LABELS)}
ID_TO_NER_LABEL = {idx: label for label, idx in NER_LABEL_TO_ID.items()}

# Lab format IDs (53 formats)
LAB_FORMATS = [
    # Australia (7)
    'australia-rcpa', 'australia-clinipath', 'australia-laverty', 'australia-sgs',
    'australia-4cyte', 'australia-apl', 'australia-qml',

    # USA (6)
    'usa-quest', 'usa-labcorp', 'usa-mayo', 'usa-arup', 'usa-kaiser', 'usa-cpmc',

    # India (5)
    'india-drlogy', 'india-lal-pathlabs', 'india-thyrocare', 'india-sterling-accuris',
    'india-dr-lal-diagnostic',

    # Southeast Asia (13)
    'singapore-general', 'singapore-parkway', 'malaysia-gribbles', 'malaysia-bp',
    'philippines-hiprecision', 'philippines-myhealth', 'indonesia-prodia', 'indonesia-pramita',
    'thailand-bangkok-hospital', 'thailand-bumrungrad', 'vietnam-vinmec', 'vietnam-family',
    'hong-kong-union',

    # Latin America (9)
    'brazil-dasa', 'brazil-fleury', 'mexico-imss', 'mexico-chopo',
    'colombia-colsanitas', 'colombia-idime', 'argentina-stamboulian', 'argentina-roffo',
    'chile-redsalud',

    # Europe (6)
    'uk-nhs', 'uk-bupa', 'germany-labor', 'france-cerba', 'spain-quiron', 'italy-synlab',

    # Canada (4)
    'canada-dynacare', 'canada-lifelabs', 'canada-gamma-dynacare', 'canada-biron',

    # Africa (1)
    'south-africa-pathcare',

    # Generic (2)
    'generic-international', 'generic-fallback',
]

FORMAT_TO_ID = {fmt: idx for idx, fmt in enumerate(LAB_FORMATS)}
ID_TO_FORMAT = {idx: fmt for fmt, idx in FORMAT_TO_ID.items()}

# Common biomarker units (200+ units)
COMMON_UNITS = [
    # Concentration
    'g/L', 'g/dL', 'mg/L', 'mg/dL', 'µg/L', 'µg/dL', 'ng/L', 'ng/dL', 'ng/mL', 'pg/mL',
    'mmol/L', 'µmol/L', 'nmol/L', 'pmol/L',

    # Cell counts
    'x10^9/L', 'x10^12/L', '10^9/L', '10^12/L', 'cells/µL', 'K/µL', 'M/µL',

    # Percentages
    '%', 'percent',

    # Enzymatic activity
    'U/L', 'IU/L', 'mU/L', 'µU/mL',

    # Ratios
    'ratio', 'index',

    # Blood gases
    'mmHg', 'kPa', 'pH',

    # Special
    'mL/min/1.73m²', 'mL/min', 's', 'sec', 'min',
]

UNIT_TO_ID = {unit: idx for idx, unit in enumerate(COMMON_UNITS)}
ID_TO_UNIT = {idx: unit for unit, idx in UNIT_TO_ID.items()}


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    model_name: str = "huawei-noah/TinyBERT_General_4L_312D"  # 14.5M params
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    ner_loss_weight: float = 1.0
    format_loss_weight: float = 0.3
    unit_loss_weight: float = 0.2
    gradient_clip: float = 1.0
    seed: int = 42


class LabReportDataset(Dataset):
    """Dataset for multi-task learning on lab reports"""

    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize text
        encoding = self.tokenizer(
            sample['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )

        # Align NER labels with tokenized input
        ner_labels = self._align_ner_labels(
            sample.get('tokens', []),
            encoding['offset_mapping'][0]
        )

        # Format classification label
        format_id = FORMAT_TO_ID.get(sample['metadata']['format'], 0)

        # Unit prediction label (predict unit from biomarker name)
        biomarker_key = sample['metadata'].get('biomarker', '')
        unit = sample['metadata'].get('unit', '')
        unit_id = UNIT_TO_ID.get(unit, 0)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'ner_labels': torch.tensor(ner_labels, dtype=torch.long),
            'format_label': torch.tensor(format_id, dtype=torch.long),
            'unit_label': torch.tensor(unit_id, dtype=torch.long),
        }

    def _align_ner_labels(self, tokens: List[Dict], offset_mapping) -> List[int]:
        """Align BIO tags with tokenizer's subword tokens"""
        labels = [NER_LABEL_TO_ID['O']] * len(offset_mapping)

        if not tokens:
            return labels

        # Create character-level label map
        char_labels = {}
        current_pos = 0
        for token_info in tokens:
            token_text = token_info['token']
            label = token_info['label']
            label_id = NER_LABEL_TO_ID.get(label, NER_LABEL_TO_ID['O'])

            # Map characters to labels
            for i in range(len(token_text)):
                char_labels[current_pos + i] = label_id

            current_pos += len(token_text) + 1  # +1 for space

        # Align with tokenized positions
        for idx, (start, end) in enumerate(offset_mapping):
            if start == end:  # Special token
                labels[idx] = NER_LABEL_TO_ID['O']
            else:
                # Use label from character position
                labels[idx] = char_labels.get(start, NER_LABEL_TO_ID['O'])

        return labels


class MultiTaskBiomarkerModel(nn.Module):
    """Multi-task model for NER + format classification + unit prediction"""

    def __init__(self, model_name: str, num_ner_labels: int, num_formats: int, num_units: int):
        super().__init__()

        # Load pretrained TinyBERT
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Task 1: NER head (token classification)
        self.ner_classifier = nn.Linear(hidden_size, num_ner_labels)

        # Task 2: Format classification head (sequence classification)
        self.format_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_formats)
        )

        # Task 3: Unit prediction head (sequence classification)
        self.unit_classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_units)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Token-level embeddings for NER
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # Sentence-level embedding for classification (CLS token)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Task outputs
        ner_logits = self.ner_classifier(sequence_output)
        format_logits = self.format_classifier(pooled_output)
        unit_logits = self.unit_classifier(pooled_output)

        return ner_logits, format_logits, unit_logits


def load_training_data(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load synthetic training data"""
    train_path = os.path.join(data_dir, 'train_samples.json')
    test_path = os.path.join(data_dir, 'test_samples.json')

    logger.info(f"Loading training data from {data_dir}")

    with open(train_path, 'r', encoding='utf-8') as f:
        train_samples = json.load(f)

    with open(test_path, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)

    logger.info(f"Loaded {len(train_samples)} training samples, {len(test_samples)} test samples")

    return train_samples, test_samples


def compute_metrics(
    ner_preds: List[List[str]],
    ner_labels: List[List[str]],
    format_preds: np.ndarray,
    format_labels: np.ndarray,
    unit_preds: np.ndarray,
    unit_labels: np.ndarray
) -> Dict:
    """Compute metrics for all three tasks"""

    # NER metrics (seqeval)
    ner_report = ner_classification_report(ner_labels, ner_preds, output_dict=True)
    ner_f1 = ner_f1_score(ner_labels, ner_preds)

    # Format classification metrics
    format_acc = accuracy_score(format_labels, format_preds)
    format_f1 = f1_score(format_labels, format_preds, average='weighted')

    # Unit prediction metrics
    unit_acc = accuracy_score(unit_labels, unit_preds)
    unit_f1 = f1_score(unit_labels, unit_preds, average='weighted')

    return {
        'ner_f1': ner_f1,
        'ner_precision': ner_report['weighted avg']['precision'],
        'ner_recall': ner_report['weighted avg']['recall'],
        'format_accuracy': format_acc,
        'format_f1': format_f1,
        'unit_accuracy': unit_acc,
        'unit_f1': unit_f1,
    }


def train_epoch(
    model: MultiTaskBiomarkerModel,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    config: TrainingConfig,
    device
) -> Dict:
    """Train for one epoch"""
    model.train()

    total_loss = 0
    ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    format_loss_fn = nn.CrossEntropyLoss()
    unit_loss_fn = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ner_labels = batch['ner_labels'].to(device)
        format_labels = batch['format_label'].to(device)
        unit_labels = batch['unit_label'].to(device)

        # Forward pass
        ner_logits, format_logits, unit_logits = model(input_ids, attention_mask)

        # Calculate losses
        # NER loss (token-level)
        ner_loss = ner_loss_fn(
            ner_logits.view(-1, len(NER_LABELS)),
            ner_labels.view(-1)
        )

        # Format classification loss
        format_loss = format_loss_fn(format_logits, format_labels)

        # Unit prediction loss
        unit_loss = unit_loss_fn(unit_logits, unit_labels)

        # Combined loss with task weights
        loss = (
            config.ner_loss_weight * ner_loss +
            config.format_loss_weight * format_loss +
            config.unit_loss_weight * unit_loss
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"Batch {batch_idx + 1}/{len(dataloader)} - "
                f"Loss: {loss.item():.4f} "
                f"(NER: {ner_loss.item():.4f}, Format: {format_loss.item():.4f}, Unit: {unit_loss.item():.4f})"
            )

    avg_loss = total_loss / len(dataloader)
    return {'loss': avg_loss}


def evaluate(
    model: MultiTaskBiomarkerModel,
    dataloader: DataLoader,
    device
) -> Dict:
    """Evaluate model on test set"""
    model.eval()

    all_ner_preds = []
    all_ner_labels = []
    all_format_preds = []
    all_format_labels = []
    all_unit_preds = []
    all_unit_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ner_labels = batch['ner_labels']
            format_labels = batch['format_label']
            unit_labels = batch['unit_label']

            # Forward pass
            ner_logits, format_logits, unit_logits = model(input_ids, attention_mask)

            # NER predictions (convert to label strings)
            ner_preds = torch.argmax(ner_logits, dim=-1).cpu().numpy()
            for i in range(len(ner_preds)):
                pred_labels = []
                true_labels = []
                for j in range(len(ner_preds[i])):
                    if ner_labels[i][j] != -100:  # Ignore padding
                        pred_labels.append(ID_TO_NER_LABEL[ner_preds[i][j]])
                        true_labels.append(ID_TO_NER_LABEL[ner_labels[i][j].item()])

                all_ner_preds.append(pred_labels)
                all_ner_labels.append(true_labels)

            # Format predictions
            format_preds = torch.argmax(format_logits, dim=-1).cpu().numpy()
            all_format_preds.extend(format_preds)
            all_format_labels.extend(format_labels.numpy())

            # Unit predictions
            unit_preds = torch.argmax(unit_logits, dim=-1).cpu().numpy()
            all_unit_preds.extend(unit_preds)
            all_unit_labels.extend(unit_labels.numpy())

    metrics = compute_metrics(
        all_ner_preds,
        all_ner_labels,
        np.array(all_format_preds),
        np.array(all_format_labels),
        np.array(all_unit_preds),
        np.array(all_unit_labels)
    )

    return metrics


def export_to_onnx(model: MultiTaskBiomarkerModel, tokenizer, output_path: str):
    """Export trained model to ONNX format for web deployment"""
    logger.info(f"Exporting model to ONNX: {output_path}")

    model.eval()
    device = next(model.parameters()).device

    # Create dummy input
    dummy_text = "Hemoglobin 150 g/L (130 - 180)"
    encoding = tokenizer(
        dummy_text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    dummy_input_ids = encoding['input_ids'].to(device)
    dummy_attention_mask = encoding['attention_mask'].to(device)

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['ner_logits', 'format_logits', 'unit_logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'ner_logits': {0: 'batch_size', 1: 'sequence_length'},
            'format_logits': {0: 'batch_size'},
            'unit_logits': {0: 'batch_size'},
        },
        opset_version=14
    )

    logger.info("ONNX export complete")


def main():
    parser = argparse.ArgumentParser(description='Train enhanced TinyBERT NER model')
    parser.add_argument('--data_dir', type=str, default='data/ml/synthetic',
                       help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='models/tinybert-ner-enhanced',
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed
    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    train_samples, test_samples = load_training_data(args.data_dir)

    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Create datasets
    train_dataset = LabReportDataset(train_samples, tokenizer, config.max_length)
    test_dataset = LabReportDataset(test_samples, tokenizer, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model
    logger.info("Initializing multi-task model")
    model = MultiTaskBiomarkerModel(
        config.model_name,
        num_ner_labels=len(NER_LABELS),
        num_formats=len(LAB_FORMATS),
        num_units=len(COMMON_UNITS)
    )
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    best_ner_f1 = 0.0

    for epoch in range(config.num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'='*60}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, config, device)
        logger.info(f"Training loss: {train_metrics['loss']:.4f}")

        # Evaluate
        eval_metrics = evaluate(model, test_loader, device)

        logger.info("\nEvaluation Metrics:")
        logger.info(f"  NER F1: {eval_metrics['ner_f1']:.4f} (Precision: {eval_metrics['ner_precision']:.4f}, Recall: {eval_metrics['ner_recall']:.4f})")
        logger.info(f"  Format Classification Accuracy: {eval_metrics['format_accuracy']:.4f} (F1: {eval_metrics['format_f1']:.4f})")
        logger.info(f"  Unit Prediction Accuracy: {eval_metrics['unit_accuracy']:.4f} (F1: {eval_metrics['unit_f1']:.4f})")

        # Save best model
        if eval_metrics['ner_f1'] > best_ner_f1:
            best_ner_f1 = eval_metrics['ner_f1']

            # Save PyTorch model
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': eval_metrics,
            }, model_path)

            logger.info(f"✅ New best model saved (NER F1: {best_ner_f1:.4f})")

    # Export to ONNX for web deployment
    onnx_path = os.path.join(args.output_dir, 'biomarker_ner_model.onnx')
    export_to_onnx(model, tokenizer, onnx_path)

    # Save metadata
    metadata = {
        'model_name': config.model_name,
        'ner_labels': NER_LABELS,
        'lab_formats': LAB_FORMATS,
        'units': COMMON_UNITS,
        'best_metrics': {
            'ner_f1': best_ner_f1,
            'final_metrics': eval_metrics
        },
        'training_config': {
            'epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'max_length': config.max_length,
        }
    }

    metadata_path = os.path.join(args.output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Training complete!")
    logger.info(f"Best NER F1 Score: {best_ner_f1:.4f}")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
