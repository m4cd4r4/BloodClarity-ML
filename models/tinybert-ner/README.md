# TinyBERT NER Model

This directory contains the model configuration files for the TinyBERT Named Entity Recognition model used in the BloodClarity-ML paper.

## Files Included

- `config.json` - Model architecture configuration
- `label_map.json` - NER label mappings (B-BIOMARKER, I-BIOMARKER, etc.)
- `tokenizer_config.json` - Tokenizer settings
- `special_tokens_map.json` - Special tokens ([CLS], [SEP], etc.)
- `tokenizer.json` - Full tokenizer configuration
- `vocab.txt` - Vocabulary file (30,522 tokens)

## Training Your Own Model

The trained model weights (`model.safetensors`, 54MB) are not included in this repository. To reproduce the results from the paper, train your own model using:

```bash
cd scripts/ml

# Generate training data
python generate_comprehensive_training_data.py --num-samples 10000

# Train the model
python train_tinybert_ner_enhanced.py

# Export to ONNX for browser deployment
python export_onnx.py --model-path ../../models/tinybert-ner/final
```

The training process takes approximately 2-3 hours on a modern GPU (RTX 3070 or better) and produces:

- `model.safetensors` (54MB) - PyTorch model weights
- `model.onnx` (54MB) - ONNX Runtime Web-compatible model

## Model Architecture

- **Base Model**: TinyBERT (14.5M parameters)
- **Task**: Named Entity Recognition (NER) for medical biomarkers
- **Labels**: 7 classes (O, B-BIOMARKER, I-BIOMARKER, B-VALUE, I-VALUE, B-UNIT, I-UNIT)
- **Max Sequence Length**: 512 tokens
- **Vocabulary Size**: 30,522 tokens

## Performance

As documented in the paper:

- **Raw ML Accuracy**: 97.6%
- **System-Level Accuracy**: 98.8% (with biological validation and context-aware units)
- **Inference Latency**: 45-80ms (browser, ONNX Runtime Web)
- **Deployment Size**: 12MB (INT8 quantised)

## Citation

If you use this model configuration in your research, please cite:

```bibtex
@article{bloodclarity2025clinical,
  title={Clinical-Grade Browser-Based Machine Learning for Medical Biomarker Extraction from Laboratory Reports},
  author={BloodClarity Research Team},
  journal={TBD},
  year={2025}
}
```

## Licence

MIT Licence - See [LICENCE](../../LICENCE) for details.
