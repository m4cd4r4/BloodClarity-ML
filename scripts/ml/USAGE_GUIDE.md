# BloodClarity Synthetic Data Generation - Complete Usage Guide

## Quick Start

### Step 1: Install Dependencies

```bash
cd C:\Scratch\bloodclarity\scripts\ml
pip install -r requirements.txt
```

### Step 2: Test the Generator

```bash
python test_generation.py
```

Expected output:
```
✓ All imports successful
✓ Loaded 30 biomarkers
...
ALL TESTS PASSED! ✓
```

### Step 3: Generate Small Test Dataset

```bash
python generate_synthetic_data.py --num-samples 100 --output "C:/Scratch/bloodclarity/data/ml/test_100.json"
```

### Step 4: Generate Full Dataset (5,000+ samples)

```bash
python generate_synthetic_data.py --num-samples 5000 --output "C:/Scratch/bloodclarity/data/ml/synthetic_lab_reports.json"
```

This will create:
- `synthetic_lab_reports.json` - Full dataset with NER labels
- `dataset_statistics.txt` - Dataset statistics and distribution

## Understanding the Output

### JSON Structure

Each sample in the output contains:

```json
{
  "id": "sample_000042",
  "text": "QUEST DIAGNOSTICS\n========================================\nReport Date: 2024-08-15\n...",
  "entities": [
    {
      "text": "Vitamin D",
      "label": "BIOMARKER_NAME",
      "start": 245,
      "end": 254
    }
  ],
  "tokens": [
    {
      "text": "Vitamin",
      "tag": "B-BIOMARKER_NAME",
      "start": 245,
      "end": 252
    }
  ],
  "format": "Quest Diagnostics",
  "unit_system": "US",
  "num_biomarkers": 8,
  "biomarkers": ["Vitamin D", "TSH", "Ferritin", "Cholesterol"]
}
```

### Entity Labels

Three types of entities are extracted:

1. **BIOMARKER_NAME**: The test name
   - Examples: "Vitamin D", "TSH", "Haemoglobin", "Creatinine"
   - Includes aliases: "25-OH Vitamin D", "Hb", "GGT"

2. **BIOMARKER_VALUE**: The numeric result
   - Examples: "45.2", "2.5", "150"
   - Realistic values (70% normal, 30% abnormal)

3. **BIOMARKER_UNIT**: The unit of measurement
   - Examples: "nmol/L", "mU/L", "g/L", "mmol/L"
   - Supports both SI and US conventional units

### BIO Tagging

The `tokens` array uses BIO (Beginning-Inside-Outside) format:

- **B-LABEL**: Beginning of an entity
- **I-LABEL**: Inside/continuation of an entity
- **O**: Outside any entity (regular text)

Example:
```
Text:     "Vitamin D: 45.2 nmol/L"
Tokens:   ["Vitamin", "D", ":", "45.2", "nmol/L"]
Tags:     ["B-BIOMARKER_NAME", "I-BIOMARKER_NAME", "O", "B-BIOMARKER_VALUE", "B-BIOMARKER_UNIT"]
```

## Advanced Usage

### Custom Biomarker Selection

Edit `generate_synthetic_data.py` to modify the biomarker selection:

```python
# Current: Random 5-15 biomarkers per report
num_biomarkers = random.randint(5, 15)

# Change to: Always 10 biomarkers
num_biomarkers = 10

# Or: Focus on specific categories
liver_markers = [b for b in biomarker_list if BIOMARKER_DATABASE[b]["category"] == "Liver"]
selected_biomarkers = random.sample(liver_markers, 5)
```

### Adjusting Out-of-Range Ratio

Modify the value generation function:

```python
# Current: 30% out of range
out_of_range = random.random() < 0.30

# Change to: 50% out of range
out_of_range = random.random() < 0.50
```

### Adding New Lab Formats

Create a new format generator function:

```python
def generate_my_custom_format(biomarkers: List[str], unit_system: str = "SI") -> Tuple[str, List[Dict]]:
    """Generate custom lab format report."""
    report_lines = []
    entities = []

    # Your format logic here
    report_lines.append("MY LAB NAME")
    # ... add biomarker lines ...

    return '\n'.join(report_lines), entities

# Add to FORMAT_GENERATORS list
FORMAT_GENERATORS.append(("My Custom Format", generate_my_custom_format, "SI"))
```

### Increasing Noise Level

Adjust OCR noise simulation:

```python
# Current: Light noise (2-3%)
display_name = add_realistic_noise(display_name, 0.02)

# Higher noise (10%)
display_name = add_realistic_noise(display_name, 0.10)
```

## Training Pipeline

### Step 1: Prepare Data Splits

```bash
python example_training_pipeline.py \
  --input "C:/Scratch/bloodclarity/data/ml/synthetic_lab_reports.json" \
  --output-dir "C:/Scratch/bloodclarity/data/ml/" \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

This creates:
- `train.json` - Training set (80%)
- `val.json` - Validation set (10%)
- `test.json` - Test set (10%)
- `label_mapping.json` - Label to ID mappings

### Step 2: Install Training Dependencies

```bash
pip install transformers torch datasets accelerate seqeval
```

### Step 3: Train TinyBERT Model

Use the example code in `example_training_pipeline.py` or create your own:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from datasets import load_dataset

# Load data
dataset = load_dataset('json', data_files={
    'train': 'data/ml/train.json',
    'validation': 'data/ml/val.json',
    'test': 'data/ml/test.json'
})

# Load TinyBERT
model_name = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=7)

# Train (see example_training_pipeline.py for full code)
trainer = Trainer(...)
trainer.train()
```

## Expected Performance

### Dataset Size Recommendations

| Use Case | Recommended Samples | Generation Time |
|----------|-------------------|-----------------|
| Quick test | 100 | ~5 seconds |
| Development | 1,000 | ~30 seconds |
| Training | 5,000 | ~2 minutes |
| Production | 10,000+ | ~5 minutes |

### Model Performance Expectations

With 5,000 training samples, expect:

- **Training time**: 30-60 minutes on GPU (few hours on CPU)
- **F1 Score**: 0.85-0.95 on validation set
- **BIOMARKER_NAME extraction**: ~95% accuracy
- **BIOMARKER_VALUE extraction**: ~90% accuracy
- **BIOMARKER_UNIT extraction**: ~92% accuracy

### File Sizes

| Dataset Size | JSON File Size | Memory Usage |
|--------------|----------------|--------------|
| 100 samples | ~200 KB | ~10 MB |
| 1,000 samples | ~2 MB | ~50 MB |
| 5,000 samples | ~10 MB | ~200 MB |
| 10,000 samples | ~20 MB | ~400 MB |

## Troubleshooting

### Issue: Import Error for faker

```
ModuleNotFoundError: No module named 'faker'
```

**Solution:**
```bash
pip install faker
```

### Issue: Permission Denied on Windows

```
PermissionError: [Errno 13] Permission denied: 'output.json'
```

**Solution:**
- Use forward slashes: `C:/path/to/file.json`
- Close any programs with the file open
- Run as administrator if needed

### Issue: Memory Error with Large Datasets

```
MemoryError: Unable to allocate array
```

**Solution:**
- Generate in batches:
```python
# Generate 10k samples in 2 batches of 5k
python generate_synthetic_data.py --num-samples 5000 --output "batch1.json"
python generate_synthetic_data.py --num-samples 5000 --output "batch2.json" --seed 99
```

### Issue: Validation Failures

```
Validation failed: Entity out of bounds
```

**Solution:**
- This is a bug in the generator
- Check entity positions match text indices
- Report issue with sample details

### Issue: Poor Model Performance

If model F1 score is below 0.80:

1. **Increase dataset size**: Generate 10,000+ samples
2. **Check data quality**: Review validation statistics
3. **Adjust noise level**: Reduce if too high
4. **Balance dataset**: Ensure even format distribution
5. **Try different model**: Test alternatives to TinyBERT

## Best Practices

### 1. Start Small

Always test with 100-1,000 samples first:
```bash
python generate_synthetic_data.py --num-samples 100 --output "test.json"
```

### 2. Use Reproducible Seeds

For consistent results:
```bash
python generate_synthetic_data.py --num-samples 5000 --seed 42
```

### 3. Validate Before Training

Check statistics file for:
- Even format distribution
- All biomarkers represented
- Reasonable entity counts

### 4. Version Your Datasets

Include generation parameters in filename:
```bash
synthetic_lab_reports_5000_seed42_v1.json
```

### 5. Monitor Training

Track these metrics:
- Training loss (should decrease)
- Validation F1 (should increase)
- Per-entity performance
- Inference time

## Integration with BloodClarity

After training, integrate model into BloodClarity:

### 1. Export Model

```python
# Save in ONNX format for faster inference
trainer.save_model("./bloodclarity-tinybert-ner")
```

### 2. Create Inference Function

```typescript
// src/utils/ml/biomarkerExtractor.ts
import { pipeline } from '@xenova/transformers';

const extractor = await pipeline('token-classification', 'bloodclarity-tinybert-ner');

export async function extractBiomarkers(text: string) {
  const results = await extractor(text);
  return parseBiomarkers(results);
}
```

### 3. Add to Upload Flow

```typescript
// src/components/UploadPage.tsx
const handleFileUpload = async (file: File) => {
  const text = await extractTextFromPDF(file);
  const biomarkers = await extractBiomarkers(text);
  // Display results...
}
```

## Additional Resources

- **Hugging Face NER Tutorial**: https://huggingface.co/docs/transformers/tasks/token_classification
- **TinyBERT Paper**: https://arxiv.org/abs/1909.10351
- **BIO Tagging Guide**: https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)
- **BloodClarity Documentation**: C:\Scratch\bloodclarity\README.md

## Support

For issues or questions:
1. Check this guide first
2. Review `README.md` in this directory
3. Check generated `dataset_statistics.txt`
4. Contact BloodClarity development team

## Version History

- **v1.0** (2025-12-10): Initial release
  - 30+ biomarkers
  - 6 lab formats
  - BIO tagging support
  - Validation and statistics
