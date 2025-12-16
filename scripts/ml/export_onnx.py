#!/usr/bin/env python3
"""
Export TinyBERT NER Model to ONNX Format
Converts the trained PyTorch model to ONNX for browser deployment
"""

import json
import os
import shutil
from pathlib import Path

import torch
import onnx
from onnx import numpy_helper
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Paths
SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR.parent.parent / "models" / "tinybert-ner"
ONNX_DIR = SCRIPT_DIR.parent.parent / "models" / "tinybert-ner-onnx"

# Export settings
MAX_LENGTH = 128
OPSET_VERSION = 14


def main():
    print("=" * 60)
    print("ONNX Export for TinyBERT NER")
    print("=" * 60)
    print()

    # Check if model exists
    if not MODEL_DIR.exists():
        print(f"ERROR: Trained model not found at {MODEL_DIR}")
        print("Please run train_tinybert_ner.py first")
        return

    # Create output directory
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    print(f"  Model loaded from: {MODEL_DIR}")
    print(f"  Labels: {model.config.num_labels}")
    print()

    # Create dummy input
    print("Creating dummy input...")
    dummy_text = "Hemoglobin 14.5 g/dL"
    tokens = dummy_text.split()

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    print(f"  Input shape: {input_ids.shape}")
    print()

    # Export to ONNX
    print("Exporting to ONNX...")
    onnx_path = ONNX_DIR / "model.onnx"

    # Define dynamic axes for variable sequence length
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size', 1: 'sequence'}
    }

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"  ONNX model saved to: {onnx_path}")
    print(f"  Model size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
    print()

    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("  Model verification passed!")
    print()

    # Test ONNX inference
    print("Testing ONNX inference...")
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))

    # Get input/output names
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    print(f"  Input names: {input_names}")
    print(f"  Output names: {output_names}")

    # Run inference
    ort_inputs = {
        'input_ids': input_ids.numpy(),
        'attention_mask': attention_mask.numpy()
    }

    ort_outputs = session.run(None, ort_inputs)
    logits = ort_outputs[0]

    print(f"  Output shape: {logits.shape}")
    print()

    # Compare with PyTorch output
    print("Comparing with PyTorch output...")
    with torch.no_grad():
        pt_outputs = model(input_ids, attention_mask)
        pt_logits = pt_outputs.logits.numpy()

    diff = abs(logits - pt_logits).max()
    print(f"  Max difference: {diff:.6f}")

    if diff < 1e-4:
        print("  Outputs match!")
    else:
        print("  WARNING: Outputs differ more than expected")
    print()

    # Save tokenizer files for browser
    print("Saving tokenizer for browser...")
    tokenizer.save_pretrained(ONNX_DIR)

    # Copy label map
    label_map_src = MODEL_DIR / "label_map.json"
    label_map_dst = ONNX_DIR / "label_map.json"
    if label_map_src.exists():
        shutil.copy(label_map_src, label_map_dst)
        print(f"  Label map copied to: {label_map_dst}")

    # Create model config for browser
    browser_config = {
        "model_type": "tinybert-ner",
        "model_name": "TinyBERT Biomarker NER",
        "max_length": MAX_LENGTH,
        "num_labels": model.config.num_labels,
        "id2label": model.config.id2label,
        "label2id": model.config.label2id,
        "onnx_file": "model.onnx",
        "tokenizer_files": [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json"
        ]
    }

    with open(ONNX_DIR / "browser_config.json", 'w') as f:
        json.dump(browser_config, f, indent=2)

    print(f"  Browser config saved to: {ONNX_DIR / 'browser_config.json'}")
    print()

    # List output files
    print("Output files:")
    print("-" * 40)
    total_size = 0
    for file in sorted(ONNX_DIR.iterdir()):
        size = file.stat().st_size
        total_size += size
        print(f"  {file.name}: {size / 1024:.1f} KB")

    print(f"\nTotal size: {total_size / (1024*1024):.2f} MB")

    print()
    print("=" * 60)
    print("ONNX EXPORT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
