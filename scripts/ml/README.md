# BloodVital ML Synthetic Data Generation

This directory contains scripts for generating synthetic lab report data to train the TinyBERT NER model for biomarker extraction.

## Overview

The `generate_synthetic_data.py` script generates realistic synthetic blood test reports across 23+ lab formats with proper NER (Named Entity Recognition) labels for training machine learning models.

## Features

- **30+ Biomarkers**: Comprehensive coverage of vitamins, thyroid, liver, kidney, lipids, haematology, and more
- **6+ Lab Formats**: Quest Diagnostics, LabCorp, NHS UK, Australian labs (Clinipath), generic formats
- **Realistic Values**: 70% in-range, 30% out-of-range with proper ranges
- **OCR Noise Simulation**: Realistic text variations and common OCR errors
- **BIO Tagging**: Proper Beginning-Inside-Outside tagging for NER training
- **Validation**: Quality checks on all generated samples

## Installation

```bash
cd C:\Scratch\bloodvital\scripts\ml
pip install -r requirements.txt
```

## Usage

### Basic Usage (Generate 5,000 samples)

```bash
python generate_synthetic_data.py
```

### Custom Number of Samples

```bash
python generate_synthetic_data.py --num-samples 10000
```

### Custom Output Location

```bash
python generate_synthetic_data.py --output "C:/path/to/output.json"
```

### With Custom Random Seed

```bash
python generate_synthetic_data.py --num-samples 5000 --seed 123
```

## Output Format

The script generates a JSON file with the following structure:

```json
[
  {
    "id": "sample_000000",
    "text": "Full report text...",
    "entities": [
      {
        "text": "Vitamin D",
        "label": "BIOMARKER_NAME",
        "start": 150,
        "end": 159
      },
      {
        "text": "45.2",
        "label": "BIOMARKER_VALUE",
        "start": 160,
        "end": 164
      },
      {
        "text": "nmol/L",
        "label": "BIOMARKER_UNIT",
        "start": 165,
        "end": 171
      }
    ],
    "tokens": [
      {
        "text": "Vitamin",
        "tag": "B-BIOMARKER_NAME",
        "start": 150,
        "end": 157
      },
      {
        "text": "D",
        "tag": "I-BIOMARKER_NAME",
        "start": 158,
        "end": 159
      }
    ],
    "format": "Quest Diagnostics",
    "unit_system": "US",
    "num_biomarkers": 8,
    "biomarkers": ["Vitamin D", "TSH", "Ferritin", ...]
  }
]
```

## Entity Labels

The script extracts three types of entities:

1. **BIOMARKER_NAME**: The name of the biomarker (e.g., "Vitamin D", "TSH", "Haemoglobin")
2. **BIOMARKER_VALUE**: The numeric value (e.g., "45.2", "2.5")
3. **BIOMARKER_UNIT**: The unit of measurement (e.g., "nmol/L", "mU/L", "g/L")

## Lab Formats

The script generates reports in these formats:

1. **Quest Diagnostics** (US) - Table format with reference ranges
2. **LabCorp** (US) - Inline format with flags
3. **NHS UK** - Structured format grouped by category
4. **Clinipath** (Australian) - Table format with Medicare info
5. **Generic Tables** - Various generic lab report styles

## Biomarker Coverage

The script includes 30+ biomarkers across these categories:

- **Vitamins**: Vitamin D, B12, Active B12, Folic Acid
- **Thyroid**: TSH
- **Inflammation**: CRP
- **Liver**: AST, ALT, Gamma GT, Alk Phos, Bilirubin, Albumin
- **Kidney**: Creatinine, eGFR, Urea
- **Electrolytes**: Sodium, Potassium, Chloride, Calcium
- **Lipids**: Cholesterol, HDL, LDL, Triglycerides
- **Haematology**: Haemoglobin, RBC, WBC, Platelets
- **Iron Studies**: Iron, Ferritin
- **Metabolic**: Glucose, HbA1c

## Realistic Features

### Value Generation
- 70% of values are within normal range
- 30% are out of range (high or low)
- Values respect proper reference ranges for each unit system (SI/US)

### OCR Noise Simulation
The script adds realistic OCR errors:
- Character substitutions (O/0, I/1/l, S/5, etc.)
- Spacing variations (extra spaces, missing spaces)
- Common text recognition errors

### Name Variations
Each biomarker can appear under multiple aliases:
- "Vitamin D" or "25-OH Vitamin D" or "Calcidiol"
- "TSH" or "Thyroid Stimulating Hormone"
- "Haemoglobin" or "Hemoglobin" or "Hb" or "Hgb"

## Validation

Each generated sample is automatically validated:
- All required fields present
- Entities within text bounds
- Valid BIO tags
- Non-empty text

## Statistics

The script automatically generates statistics:
- Total samples
- Format distribution
- Unit system distribution
- Biomarker frequency
- Average biomarkers per report

Statistics are saved to `dataset_statistics.txt` in the same directory as the output file.

## Example Output Statistics

```
SYNTHETIC DATASET STATISTICS
======================================================================

Total Samples: 5000
Average Biomarkers per Report: 9.85
Total Entities: 147500
Total Tokens: 625000

Format Distribution:
  Quest Diagnostics: 850 (17.0%)
  LabCorp: 835 (16.7%)
  NHS UK: 840 (16.8%)
  Clinipath AU: 825 (16.5%)
  Generic Table 1: 830 (16.6%)
  Generic Table 2: 820 (16.4%)

Unit System Distribution:
  SI: 3325 (66.5%)
  US: 1675 (33.5%)
```

## Next Steps

After generating the synthetic data:

1. **Split the data**: Create train/validation/test splits (e.g., 80/10/10)
2. **Convert to Hugging Face format**: Use the `datasets` library
3. **Train TinyBERT**: Fine-tune for NER task
4. **Evaluate**: Test on real lab reports
5. **Iterate**: Adjust generation parameters based on model performance

## Troubleshooting

### Import Errors
```bash
# Make sure faker is installed
pip install faker
```

### Path Issues
- Use forward slashes in paths: `C:/path/to/file.json`
- Or use raw strings: `r"C:\path\to\file.json"`

### Memory Issues
If generating very large datasets (50k+ samples):
- Generate in batches
- Write to file incrementally
- Use `--num-samples` to control size

## Contact

For questions or issues, contact the BloodVital team.
