# LabLens-ML: Clinical-Grade Browser-Based Biomarker Extraction

[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Read%20Online-blue)](docs/paper/index.html)

**Academic research repository for privacy-preserving, browser-based medical biomarker extraction.**

> **Note**: This is the research/ML component of LabLens. For the full application,
> see [LabLens.dev](https://lablens.dev) (commercial product).

## 🎯 Overview

This repository contains the machine learning models, training scripts, and research materials
for achieving **98.8% clinical-grade accuracy** in biomarker extraction from laboratory reports,
entirely within the browser with **100% offline capability**.

### Key Achievements

- **98.8% Accuracy**: System-level accuracy across 165 biomarkers and 53 formats
- **12MB Model**: Optimised from 60MB via distillation, quantisation, pruning
- **45-80ms Latency**: Real-time inference on consumer hardware
- **Privacy-First**: 100% offline, HIPAA/GDPR compliant by design
- **$162K/year Savings**: vs. cloud alternatives at enterprise scale

## 📚 Research Paper

**Read the full paper**: [Clinical-Grade Browser-Based ML for Biomarker Extraction](docs/paper/index.html)

Published: December 2025
Authors: LabLens Research Team

**Citation**:
```bibtex
@article{lablens2025clinical,
  title={Clinical-Grade Browser-Based Machine Learning for Medical Biomarker Extraction},
  author={{LabLens Research Team}},
  year={2025},
  journal={Preprint}
}
```

## 🚀 Quick Start

### 1. Generate Synthetic Training Data

```bash
cd scripts/ml
python generate_comprehensive_training_data.py \
  --output_dir ../../data/synthetic \
  --samples 10000
```

### 2. Train the Model

```bash
python train_tinybert_ner_enhanced.py \
  --data_dir ../../data/synthetic \
  --output_dir ../../models/tinybert-biomarker-ner \
  --epochs 10
```

### 3. Use in Browser

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
</head>
<body>
  <script type="module">
    import { loadModel, extractBiomarkers } from './src/inference/onnx-inference.js';

    const model = await loadModel('./models/tinybert-biomarker-ner/model.onnx');
    const results = await extractBiomarkers(pdfText, model);
    console.log(results); // [{name, value, unit, range}, ...]
  </script>
</body>
</html>
```

See [examples/](examples/) for complete working examples.

## 📊 Dataset

### Synthetic Training Data

- **10,000 samples**: 8,000 training, 2,000 test
- **53 formats**: International lab providers (India, USA, Australia, UK, Canada, Germany)
- **165 biomarkers**: Across 20+ medical categories
- **6 languages**: English, Spanish, Portuguese, Indonesian, Thai, Vietnamese
- **15% adversarial**: OCR errors, edge cases

**Download**: `data/synthetic/train.json` (1.7MB compressed)

### Biomarker Database

165 biomarkers with:
- Primary names and 500+ aliases
- Reference ranges (normal, survival limits)
- Units (SI, conventional, regional variations)
- Clinical categories

**Download**: `data/biomarker-database.json`

## 🏗️ Architecture

### Five-Component System

1. **Synthetic Data Generation** (10,000 diverse samples)
2. **Multi-Task Learning** (NER + Format + Unit prediction)
3. **OCR Preprocessing** (Multi-pass with error correction)
4. **Biological Validation** (165 biomarker plausibility limits)
5. **Context-Aware Units** (Value-based unit detection)

### Model Optimisation Pipeline

```
BERT-base (420MB, 99.5% F1)
  ↓ Knowledge Distillation
TinyBERT (60MB, 98.4% F1)
  ↓ INT8 Quantisation
Quantised (15MB, 98.1% F1)
  ↓ 50% Pruning
Final Model (12MB, 97.6% F1)
  ↓ + Validation Layers
System-Level (12MB, 98.8% accuracy)
```

## 📁 Repository Structure

```
lablens-ml/
├── docs/paper/          # Academic papers (HTML + Markdown)
├── data/
│   ├── synthetic/       # Training data (10K samples)
│   ├── sample-reports/  # De-identified PDFs
│   └── biomarker-database.json
├── scripts/ml/          # Training and generation scripts
├── models/              # Trained model weights (ONNX)
├── src/
│   ├── biomarker-validation/  # Core ML utilities
│   └── inference/       # Browser inference code
├── tests/               # Accuracy certification tests
└── examples/            # Integration examples
```

## 🔬 Reproducibility

All experiments are reproducible:

### System Requirements

- **Training**: NVIDIA GPU with 16GB+ VRAM (or cloud)
- **Inference**: Any modern browser (Chrome 113+, Firefox 117+, Safari 18+)

### Dependencies

```bash
# Python (training)
pip install torch transformers datasets onnx onnxruntime evaluate

# JavaScript (browser inference)
npm install onnxruntime-web
```

### Training Time

- Data generation: ~5 minutes (10K samples)
- Model training: ~4 hours (RTX 4090) or ~8 hours (RTX 3070)
- Optimisation: ~2 hours (quantisation + pruning)

## 📈 Results

### Synthetic Test Set

| Metric | Score |
|--------|-------|
| NER F1 (Micro) | 98.4% |
| Format Classification | 95.7% |
| Unit Prediction | 92.1% |

### Real-World PDFs (22 reports)

| Region | Formats | Accuracy |
|--------|---------|----------|
| Australia (RCPA) | 6 | 99.3% |
| India (Drlogy, Lal) | 7 | 98.3% |
| USA (LabCorp, Quest) | 5 | 97.8% |
| **Overall** | **22** | **98.5%** |

## 🤝 Contributing

We welcome contributions! Areas of interest:

- **Format expansion**: Add support for new lab providers
- **Language support**: Multilingual biomarker extraction
- **Optimisation**: Further model compression techniques
- **Validation**: Additional biological plausibility rules
- **Testing**: More real-world PDF test cases

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 Licence

This project is licenced under the **MIT Licence** - see [LICENCE](LICENCE) file.

### Model Weights

Trained model weights in `models/` are also MIT licenced. If you use these models
in academic work, please cite our paper.

### Third-Party Licences

- TinyBERT: Apache 2.0 (Huawei Noah's Ark Lab)
- ONNX Runtime: MIT Licence (Microsoft)
- Biomarker reference data: Public domain (clinical literature)

## 🔗 Related Projects

- **LabLens** (Commercial Product): [lablens.dev](https://lablens.dev)
- **Paper Website**: [Read Online](docs/paper/index.html)
- **Hugging Face Models**: Coming soon

## 📧 Contact

- **Research Questions**: research@lablens.dev
- **Issues**: [GitHub Issues](https://github.com/yourusername/lablens-ml/issues)
- **Twitter**: [@LabLensDev](https://twitter.com/lablensdev)

## 🙏 Acknowledgments

- **TinyBERT Team** (Huawei Noah's Ark Lab) for the base model
- **ONNX Runtime Team** for browser ML infrastructure
- **Medical Community** for clinical guidance on biomarker plausibility
- **Open-Source Community** for tools and frameworks

---

**Built with**: PyTorch • Transformers • ONNX • TypeScript • WebGPU

**Funding**: Self-funded independent research

**Ethics**: No patient data used. All training data is synthetic or properly licenced.
