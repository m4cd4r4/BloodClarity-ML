# BloodClarity ML - Quick Reference Card

## Installation (One-time)

```bash
cd C:\Scratch\bloodclarity\scripts\ml
pip install -r requirements.txt
```

## Generate Data

```bash
# Test generation (10 samples)
python generate_synthetic_data.py --num-samples 10 --output "test.json"

# Full dataset (5,000 samples)
python generate_synthetic_data.py --num-samples 5000

# Large dataset (10,000 samples)
python generate_synthetic_data.py --num-samples 10000 --output "C:/Scratch/bloodclarity/data/ml/large_dataset.json"
```

## Test Generation

```bash
# Quick test
python test_generation.py
```

## Explore Data

```bash
# Interactive explorer
python explore_data.py --input "synthetic_lab_reports.json"

# Show random samples
python explore_data.py --input "synthetic_lab_reports.json" --mode sample --num-samples 5

# Show statistics
python explore_data.py --input "synthetic_lab_reports.json" --mode stats

# Quality check
python explore_data.py --input "synthetic_lab_reports.json" --mode quality
```

## Prepare for Training

```bash
# Split into train/val/test
python example_training_pipeline.py \
  --input "synthetic_lab_reports.json" \
  --output-dir "C:/Scratch/bloodclarity/data/ml/" \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

## File Outputs

| File | Description |
|------|-------------|
| `synthetic_lab_reports.json` | Full dataset with NER labels |
| `dataset_statistics.txt` | Dataset statistics |
| `train.json` | Training split (80%) |
| `val.json` | Validation split (10%) |
| `test.json` | Test split (10%) |
| `label_mapping.json` | Label to ID mappings |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-samples` | 5000 | Number of samples to generate |
| `--output` | Default path | Output JSON file |
| `--seed` | 42 | Random seed for reproducibility |

## Entity Labels

- **BIOMARKER_NAME**: Test name (e.g., "Vitamin D", "TSH")
- **BIOMARKER_VALUE**: Numeric result (e.g., "45.2", "2.5")
- **BIOMARKER_UNIT**: Unit (e.g., "nmol/L", "mU/L")

## BIO Tags

- **B-LABEL**: Beginning of entity
- **I-LABEL**: Inside entity
- **O**: Outside entity (normal text)

## Lab Formats

1. Quest Diagnostics (US)
2. LabCorp (US)
3. NHS UK
4. Clinipath (Australian)
5. Generic Table (SI units)
6. Generic Table (US units)

## Biomarker Categories

- Vitamins (4 markers)
- Thyroid (1 marker)
- Inflammation (1 marker)
- Liver (6 markers)
- Kidney (3 markers)
- Electrolytes (5 markers)
- Lipids (5 markers)
- Haematology (6 markers)
- Iron Studies (2 markers)
- Metabolic (2 markers)

**Total: 30+ biomarkers**

## Common Issues

| Issue | Solution |
|-------|----------|
| Import error | `pip install faker` |
| Permission denied | Use forward slashes in paths |
| Memory error | Generate in smaller batches |
| Validation fails | Check entity positions |

## Expected Performance

| Dataset Size | Generation Time | File Size |
|--------------|-----------------|-----------|
| 100 samples | ~5 seconds | ~200 KB |
| 1,000 samples | ~30 seconds | ~2 MB |
| 5,000 samples | ~2 minutes | ~10 MB |
| 10,000 samples | ~5 minutes | ~20 MB |

## Model Training Expectations

- **Training time**: 30-60 min (GPU) / 2-4 hours (CPU)
- **Expected F1**: 0.85-0.95
- **Best model**: TinyBERT (fast, accurate)
- **Alternative**: BERT-base, RoBERTa

## Quick Commands

```bash
# Generate and explore in one go
python generate_synthetic_data.py --num-samples 1000 && \
python explore_data.py --input "C:/Scratch/bloodclarity/data/ml/synthetic_lab_reports.json" --mode stats

# Full pipeline
python generate_synthetic_data.py --num-samples 5000 && \
python example_training_pipeline.py --input "C:/Scratch/bloodclarity/data/ml/synthetic_lab_reports.json"
```

## Help Commands

```bash
python generate_synthetic_data.py --help
python explore_data.py --help
python example_training_pipeline.py --help
```

## Documentation Files

- `README.md` - Complete overview
- `USAGE_GUIDE.md` - Detailed usage instructions
- `QUICK_REFERENCE.md` - This file

## Next Steps After Generation

1. Run quality check: `python explore_data.py --mode quality`
2. Review statistics: Check `dataset_statistics.txt`
3. Prepare splits: Run `example_training_pipeline.py`
4. Install training deps: `pip install transformers torch datasets`
5. Train model: Follow training pipeline code
6. Integrate into BloodClarity: Add to upload flow

---

**Last Updated**: 2025-12-10
**Version**: 1.0
