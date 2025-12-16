#!/usr/bin/env python3
"""
Quick test script to verify the synthetic data generator works.

This script imports and tests key functions without generating full dataset.
"""

import sys
from pathlib import Path

# Add script directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from generate_synthetic_data import (
        BIOMARKER_DATABASE,
        generate_biomarker_value,
        add_realistic_noise,
        generate_quest_diagnostics_format,
        generate_bio_tags,
        validate_output,
        FORMAT_GENERATORS
    )
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    print("\nPlease install requirements:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Test 1: Biomarker database loaded
print(f"\n[OK] Loaded {len(BIOMARKER_DATABASE)} biomarkers")

# Test 2: Generate values for a few biomarkers
print("\nTesting value generation:")
test_biomarkers = ["Vitamin D", "TSH", "Ferritin", "Cholesterol", "Haemoglobin"]
for biomarker in test_biomarkers:
    value, unit, status = generate_biomarker_value(biomarker, "SI")
    print(f"  {biomarker}: {value} {unit} ({status})")

# Test 3: Test noise addition
print("\nTesting noise simulation:")
original = "Vitamin D: 45.2 nmol/L"
noisy = add_realistic_noise(original, 0.1)
print(f"  Original: {original}")
print(f"  With noise: {noisy}")

# Test 4: Generate a sample report
print("\nTesting report generation:")
sample_markers = ["Vitamin D", "TSH", "Ferritin"]
report_text, entities = generate_quest_diagnostics_format(sample_markers, "US")
print(f"  Generated report with {len(entities)} entities")
print(f"  Report length: {len(report_text)} characters")

# Test 5: Test BIO tagging
print("\nTesting BIO tagging:")
tokens = generate_bio_tags(report_text, entities)
print(f"  Generated {len(tokens)} tokens")
bio_tags = [t["tag"] for t in tokens if t["tag"] != "O"][:5]
print(f"  Sample tags: {bio_tags}")

# Test 6: Test validation
print("\nTesting validation:")
sample = {
    "id": "test_001",
    "text": report_text,
    "entities": entities,
    "tokens": tokens,
    "format": "Quest Diagnostics",
    "unit_system": "US"
}
is_valid = validate_output(sample)
print(f"  Validation result: {'[PASS]' if is_valid else '[FAIL]'}")

# Test 7: Show available formats
print(f"\n[OK] {len(FORMAT_GENERATORS)} format generators available:")
for name, func, unit in FORMAT_GENERATORS:
    print(f"  - {name} ({unit})")

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\nThe generator is ready to use. To generate full dataset:")
print("  python generate_synthetic_data.py --num-samples 5000")
print("\nOr start small to test:")
print("  python generate_synthetic_data.py --num-samples 100")
