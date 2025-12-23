#!/usr/bin/env python3
"""
Convert trained TinyBERT NER model to ONNX format for browser deployment.

The ONNX model can be used with:
- ONNX Runtime Web (onnxruntime-web)
- Transformers.js (@xenova/transformers)

Both options work great in browsers with WebAssembly or WebGPU acceleration.
"""

import json
import os
import shutil
from pathlib import Path

print("=" * 70)
print("BloodClarity Model Conversion: PyTorch -> ONNX")
print("=" * 70)

# Check for required packages
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    print("[OK] PyTorch and Transformers loaded")
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    exit(1)

try:
    from optimum.onnxruntime import ORTModelForTokenClassification
    print("[OK] Optimum ONNX Runtime loaded")
    HAS_OPTIMUM = True
except ImportError:
    print("[WARN] Optimum not installed. Install with: pip install optimum[onnxruntime]")
    HAS_OPTIMUM = False

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = BASE_DIR / "models" / "tinybert-ner" / "final"
ONNX_DIR = BASE_DIR / "models" / "tinybert-ner-onnx"
PUBLIC_MODEL_DIR = BASE_DIR / "public" / "models" / "biomarker-ner"

print(f"\nSource model: {MODEL_DIR}")
print(f"ONNX output: {ONNX_DIR}")
print(f"Public output: {PUBLIC_MODEL_DIR}")

# Check source model exists
if not MODEL_DIR.exists():
    print(f"\n[ERROR] Source model not found at {MODEL_DIR}")
    print("Run train_model.py first to train the model.")
    exit(1)

# Load the model
print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))
print(f"[OK] Loaded model with {model.num_parameters():,} parameters")

# Get label mapping
id2label = model.config.id2label
label2id = model.config.label2id
print(f"[OK] Labels: {list(label2id.keys())}")

if HAS_OPTIMUM:
    # Convert to ONNX using Optimum (recommended approach)
    print("\n" + "=" * 70)
    print("CONVERTING TO ONNX (using Optimum)")
    print("=" * 70)

    # Create output directory
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    ort_model = ORTModelForTokenClassification.from_pretrained(
        str(MODEL_DIR),
        export=True
    )

    # Save the ONNX model
    ort_model.save_pretrained(str(ONNX_DIR))
    tokenizer.save_pretrained(str(ONNX_DIR))

    print(f"[OK] ONNX model saved to: {ONNX_DIR}")

else:
    # Fallback: Manual ONNX export
    print("\n" + "=" * 70)
    print("CONVERTING TO ONNX (manual export)")
    print("=" * 70)

    try:
        import onnx
        print("[OK] ONNX loaded")
    except ImportError:
        print("[ERROR] ONNX not installed. Install with: pip install onnx")
        exit(1)

    # Create output directory
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    # Create dummy input
    dummy_input = tokenizer(
        "Vitamin D: 45.2 nmol/L",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True
    )

    # Export to ONNX
    onnx_path = ONNX_DIR / "model.onnx"

    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size", 1: "sequence"}
        },
        opset_version=14
    )

    # Also save tokenizer and config
    tokenizer.save_pretrained(str(ONNX_DIR))
    model.config.save_pretrained(str(ONNX_DIR))

    print(f"[OK] ONNX model saved to: {onnx_path}")

# Copy to public folder for web access
print("\n" + "=" * 70)
print("COPYING TO PUBLIC FOLDER")
print("=" * 70)

PUBLIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Copy essential files
essential_files = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json"
]

# Copy ONNX model file(s)
for f in ONNX_DIR.glob("*.onnx"):
    shutil.copy(f, PUBLIC_MODEL_DIR / f.name)
    print(f"  Copied: {f.name}")

# Copy tokenizer files
for f in essential_files:
    src = ONNX_DIR / f
    if src.exists():
        shutil.copy(src, PUBLIC_MODEL_DIR / f)
        print(f"  Copied: {f}")

# Create a model info file for the frontend
model_info = {
    "name": "biomarker-ner",
    "description": "TinyBERT NER model for biomarker extraction from lab reports",
    "version": "1.0.0",
    "labels": list(label2id.keys()),
    "label2id": label2id,
    "id2label": {str(k): v for k, v in id2label.items()},
    "maxLength": 256,
    "format": "onnx"
}

with open(PUBLIC_MODEL_DIR / "model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)
print("  Created: model_info.json")

# List output files
print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

total_size = 0
for f in sorted(PUBLIC_MODEL_DIR.iterdir()):
    size = f.stat().st_size
    total_size += size
    size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
    print(f"  {f.name}: {size_str}")

print(f"\nTotal size: {total_size / 1024 / 1024:.1f} MB")

print("\n" + "=" * 70)
print("CONVERSION COMPLETE!")
print("=" * 70)
print(f"\nModel ready for browser deployment at:")
print(f"  {PUBLIC_MODEL_DIR}")
print("\nNext steps:")
print("  1. Install ONNX Runtime Web: npm install onnxruntime-web")
print("  2. Or use Transformers.js: npm install @xenova/transformers")
print("  3. Load model in browser and run inference")
