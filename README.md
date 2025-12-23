# BloodClarity ML - Browser-Based Medical NER

[![Licence: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Licence: CC BY 4.0](https://img.shields.io/badge/Paper-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/Paper-Read%20Online-blue)](paper/CLINICAL-GRADE-BROWSER-ML-FOR-BIOMARKER-EXTRACTION.md)

**Privacy-preserving, browser-based biomarker extraction from laboratory reports.**

## Research Paper

📄 **[High-Accuracy Browser-Based Machine Learning for Medical Biomarker Extraction](paper/CLINICAL-GRADE-BROWSER-ML-FOR-BIOMARKER-EXTRACTION.md)**

A five-component system achieving **98.8% extraction accuracy** (n=22 international PDFs, 687 biomarkers) with complete offline capability:

1. Synthetic training data generation (10,000 samples, 16 formats, 102 biomarkers)
2. Multi-task learning architecture (NER + format + unit prediction)
3. Multi-pass OCR preprocessing with error correction
4. Biological plausibility validation
5. Context-aware unit conversion

**Author:** Macdara Ó Murchú
**Date:** December 2025

*This is a living document. Content will evolve as R&D progresses.*

## Key Results

| Metric | Value |
|--------|-------|
| System Accuracy | 98.8% |
| Model Size | 12MB (optimised) |
| Inference Latency | 45-80ms |
| Offline Capability | 100% |
| Biomarkers Supported | 102 |
| Lab Formats | 16 international |

## Repository Contents

```
bloodclarity-ml/
├── paper/               # Academic paper (Markdown)
├── data/synthetic/      # Training data (10K samples)
├── models/tinybert-ner/ # Model configuration
├── scripts/ml/          # Training pipeline
└── src/                 # Validation utilities
```

## Quick Start

### Use Pre-trained Model

```bash
# Download model (55MB ONNX format)
wget https://github.com/m4cd4r4/bloodclarity-ml/releases/download/v1.0.0/model.onnx \
  -O models/tinybert-ner-onnx/model.onnx
```

### Train From Scratch

```bash
cd scripts/ml

# Generate synthetic training data
python generate_comprehensive_training_data.py --num-samples 10000

# Train TinyBERT model
python train_tinybert_ner_enhanced.py --epochs 10

# Export to ONNX
python export_onnx.py
```

See [scripts/ml/README.md](scripts/ml/README.md) for detailed instructions.

## Citation

```bibtex
@software{bloodclarity_ml_2025,
  author = {Macdara Ó Murchú},
  title = {BloodClarity-ML: High-Accuracy Browser-Based Biomarker Extraction},
  year = {2025},
  url = {https://github.com/m4cd4r4/bloodclarity-ml},
  version = {1.0.0}
}
```

## Licence

- **Paper**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) - Creative Commons Attribution
- **Code & Models**: [MIT](LICENCE) - Open source

## Related

- **BloodClarity** (Application): [bloodclarity.com](https://bloodclarity.com)
- **Issues**: [GitHub Issues](https://github.com/m4cd4r4/bloodclarity-ml/issues)

---

**Built with**: PyTorch • Transformers • ONNX • TypeScript • WebGPU
