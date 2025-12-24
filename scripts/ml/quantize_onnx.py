#!/usr/bin/env python3
"""
INT8 Quantization for TinyBERT NER ONNX Model

Reduces model size from ~54MB to ~12MB using dynamic INT8 quantization.
This is ideal for browser deployment where smaller models load faster.

Usage:
    python quantize_onnx.py [--input MODEL_PATH] [--output OUTPUT_PATH]
"""

import argparse
from pathlib import Path
import shutil

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx
except ImportError:
    print("ERROR: Missing dependencies. Install with:")
    print("  pip install onnxruntime onnx")
    exit(1)

# Default paths
SCRIPT_DIR = Path(__file__).parent
ONNX_DIR = SCRIPT_DIR.parent.parent / "models" / "tinybert-ner-onnx"
DEFAULT_INPUT = ONNX_DIR / "model.onnx"
DEFAULT_OUTPUT = ONNX_DIR / "model_int8.onnx"


def quantize_model(input_path: Path, output_path: Path, weight_type: QuantType = QuantType.QInt8):
    """
    Apply dynamic INT8 quantization to reduce model size.

    Dynamic quantization quantizes weights to INT8 at export time,
    but computes activations in FP32 at runtime. This provides
    good compression (4x) with minimal accuracy loss.

    Args:
        input_path: Path to FP32 ONNX model
        output_path: Path for quantized output
        weight_type: Quantization type (default QInt8)
    """
    print(f"Input model: {input_path}")
    print(f"Output model: {output_path}")
    print(f"Quantization type: {weight_type}")
    print()

    # Check input exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    # Get original size
    original_size = input_path.stat().st_size
    print(f"Original model size: {original_size / (1024*1024):.2f} MB")

    # Apply dynamic quantization
    print("\nApplying dynamic INT8 quantization...")
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=weight_type,
        per_channel=False,  # Per-tensor quantization (simpler, compatible)
        reduce_range=False,  # Full INT8 range
    )

    # Verify output
    print("Verifying quantized model...")
    quantized_model = onnx.load(str(output_path))
    onnx.checker.check_model(quantized_model)

    # Get quantized size
    quantized_size = output_path.stat().st_size
    compression_ratio = original_size / quantized_size

    print()
    print("=" * 50)
    print("QUANTIZATION RESULTS")
    print("=" * 50)
    print(f"Original size:   {original_size / (1024*1024):>8.2f} MB")
    print(f"Quantized size:  {quantized_size / (1024*1024):>8.2f} MB")
    print(f"Compression:     {compression_ratio:>8.1f}x")
    print(f"Size reduction:  {(1 - quantized_size/original_size) * 100:>7.1f}%")
    print("=" * 50)

    return output_path


def test_quantized_model(model_path: Path):
    """Test that quantized model produces valid outputs."""
    import onnxruntime as ort
    import numpy as np

    print(f"\nTesting quantized model: {model_path}")

    # Create session
    session = ort.InferenceSession(str(model_path))

    # Get input info
    inputs = session.get_inputs()
    print(f"Inputs: {[inp.name for inp in inputs]}")

    # Create dummy input (batch=1, seq=128)
    dummy_input_ids = np.ones((1, 128), dtype=np.int64)
    dummy_attention_mask = np.ones((1, 128), dtype=np.int64)

    # Run inference
    outputs = session.run(None, {
        'input_ids': dummy_input_ids,
        'attention_mask': dummy_attention_mask
    })

    logits = outputs[0]
    print(f"Output shape: {logits.shape}")
    print(f"Output dtype: {logits.dtype}")
    print(f"Output range: [{logits.min():.4f}, {logits.max():.4f}]")

    # Check for NaN/Inf
    if np.isnan(logits).any() or np.isinf(logits).any():
        print("WARNING: Output contains NaN or Inf values!")
        return False

    print("Model test PASSED!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input FP32 ONNX model (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output quantized model (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace original model.onnx with quantized version"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=True,
        help="Test quantized model after creation (default: True)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("INT8 QUANTIZATION FOR TINYBERT NER")
    print("=" * 60)
    print()

    # Quantize
    output_path = quantize_model(args.input, args.output)

    # Test
    if args.test:
        test_quantized_model(output_path)

    # Replace original if requested
    if args.replace:
        print(f"\nReplacing original model with quantized version...")
        backup_path = args.input.with_suffix('.onnx.fp32.bak')
        shutil.move(args.input, backup_path)
        shutil.move(output_path, args.input)
        print(f"  Original backed up to: {backup_path}")
        print(f"  Quantized model now at: {args.input}")

    print()
    print("=" * 60)
    print("QUANTIZATION COMPLETE")
    print("=" * 60)
    print()
    print("To deploy the smaller model to the app:")
    print(f"  cp {output_path} /path/to/lablens/public/models/biomarker-ner/model.onnx")


if __name__ == "__main__":
    main()
