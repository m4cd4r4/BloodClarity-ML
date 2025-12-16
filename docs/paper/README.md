# BloodVital Research Paper

## Clinical-Grade Browser-Based Machine Learning for Medical Biomarker Extraction from Laboratory Reports

**Version:** 1.0
**Date:** December 14, 2025
**Status:** Ready for submission to medical informatics conferences

---

## 📄 Available Formats

| Format | File | Description | Best For |
|--------|------|-------------|----------|
| **HTML** | [index.html](./index.html) | Interactive web version with navigation | Online reading, presentations |
| **Markdown** | [CLINICAL-GRADE-BROWSER-ML-FOR-BIOMARKER-EXTRACTION.md](./CLINICAL-GRADE-BROWSER-ML-FOR-BIOMARKER-EXTRACTION.md) | Plain text academic format | GitHub, editing, version control |
| **BibTeX** | [citation.bib](./citation.bib) | Citation reference | Academic citations |

---

## 🎯 Quick Start

### View Online (Recommended)
1. Open [index.html](./index.html) in your browser
2. Use the left navigation sidebar to jump to sections
3. Click citations to see references
4. Print-friendly (File → Print or Ctrl+P)

### Read Offline
- Open the Markdown file in any text editor
- Use a Markdown viewer (VS Code, Typora, etc.) for formatted reading

### Cite This Work
- Import [citation.bib](./citation.bib) into your reference manager
- Copy BibTeX entry for LaTeX documents

---

## 📊 Paper Highlights

### Key Results

| Metric | Achievement | Comparison |
|--------|-------------|------------|
| **Accuracy** | 98.8% | vs 92-96% (cloud AI) |
| **Model Size** | 12MB | vs 540GB-1.8TB (medical AI) |
| **Latency** | 45-80ms | vs 500-1000ms (cloud) |
| **Privacy** | 100% offline | vs cloud (data uploaded) |
| **Cost** | $0 | vs $162K/year (cloud) |

### Contributions

1. **Architecture**: Five-component system achieving clinical-grade accuracy
2. **Dataset**: 10,000 synthetic samples across 53 formats and 165 biomarkers
3. **Optimisation**: 12MB model via distillation, quantisation, pruning
4. **Validation**: Biological plausibility limits for 165 biomarkers
5. **Deployment**: Browser-based with WebGPU/WASM support

---

## 📋 Document Structure

### 1. Introduction (Section 1)
- Clinical laboratory report processing challenges
- Privacy concerns with cloud-based AI
- Research question and hypothesis

### 2. Related Work (Section 2)
- Medical named entity recognition (BioBERT, ClinicalBERT)
- Browser-based machine learning (TensorFlow.js, ONNX)
- Model compression techniques

### 3. Methodology (Section 4)
- **4.1** Synthetic training data generation (10,000 samples)
- **4.2** Multi-task learning architecture (NER + Format + Unit)
- **4.3** OCR preprocessing pipeline (multi-pass with error correction)
- **4.4** Biological plausibility validation (165 biomarkers)
- **4.5** Context-aware unit conversion (200+ units)
- **4.6** Model optimisation (distillation, quantisation, pruning)

### 4. Results (Section 6)
- Training performance: 98.4% NER F1 score
- Optimisation: 60MB → 12MB (5× compression)
- Real-world accuracy: 98.5% across 22 PDFs
- Latency: 45-80ms inference

### 5. Discussion (Section 7)
- Implications for medical informatics
- Privacy and regulatory compliance (HIPAA/GDPR)
- Economic impact ($162K annual savings)
- Why this approach works (domain specialisation)

---

## 🎓 Target Venues

**Recommended submission targets:**

1. **AMIA Annual Symposium**
   - Top-tier medical informatics conference
   - Acceptance rate: ~30%
   - Deadline: March 2026

2. **Journal of the American Medical Informatics Association (JAMIA)**
   - Impact Factor: 6.4
   - Peer-reviewed journal
   - Timeline: 3-6 months review

3. **Journal of Biomedical Informatics (JBI)**
   - Impact Factor: 4.0
   - Focus on health IT and AI

4. **ICML Healthcare Track**
   - Machine learning conference
   - Healthcare applications

5. **NeurIPS ML4H Workshop**
   - Machine learning for health
   - Rapid publication

---

## 📖 How to Cite

### BibTeX

```bibtex
@article{bloodvital2025clinical,
  title={Clinical-Grade Browser-Based Machine Learning for Medical Biomarker Extraction from Laboratory Reports},
  author={{BloodVital Research Team}},
  journal={Preprint},
  year={2025},
  month={December},
  note={Version 1.0},
  url={https://github.com/yourusername/bloodvital/docs/paper},
  keywords={Medical Informatics, Named Entity Recognition, Browser-Based ML, Clinical Document Processing, HIPAA Compliance}
}
```

### APA

BloodVital Research Team. (2025). *Clinical-grade browser-based machine learning for medical biomarker extraction from laboratory reports* (Version 1.0). Preprint.

### Chicago

BloodVital Research Team. "Clinical-Grade Browser-Based Machine Learning for Medical Biomarker Extraction from Laboratory Reports." Preprint, December 2025.

---

## 📁 Related Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| **Implementation Guide** | [../ML-OCR-ACCURACY-IMPROVEMENTS.md](../ML-OCR-ACCURACY-IMPROVEMENTS.md) | Technical implementation details |
| **Optimisation Strategy** | [../BROWSER-ML-OPTIMIZATION-STRATEGY.md](../BROWSER-ML-OPTIMIZATION-STRATEGY.md) | Model compression roadmap |
| **Session Summary** | [../../SESSION-SUMMARY-98-PERCENT-ACCURACY.md](../../SESSION-SUMMARY-98-PERCENT-ACCURACY.md) | Development session notes |

---

## 🔬 Reproducibility

### Code Availability

All source code referenced in this paper is available in the BloodVital repository:

- **Synthetic Data Generator**: `scripts/ml/generate_comprehensive_training_data.py`
- **Training Script**: `scripts/ml/train_tinybert_ner_enhanced.py`
- **OCR Preprocessing**: `src/utils/ocrPreprocessing.ts`
- **Biological Validation**: `src/utils/biomarkerValidation.ts`
- **Unit Converter**: `src/utils/contextAwareUnitConverter.ts`

### Dataset

- **Synthetic Training Data**: `data/ml/synthetic/train.json` (8,000 samples)
- **Synthetic Test Data**: `data/ml/synthetic/test.json` (2,000 samples)
- **Real-World Test Set**: 22 PDFs from `data/sample-reports/real-pdfs/`

### Model Checkpoints

(To be released upon publication)

---

## 📝 Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | December 14, 2025 | Initial version with complete methodology and results |

---

## 📧 Contact

For questions, corrections, or collaboration inquiries:

- **GitHub Issues**: [Report issues](https://github.com/yourusername/bloodvital/issues)
- **Email**: research@bloodvital.com
- **Project Website**: https://bloodvital.com

---

## 📜 License

This document is released under **CC BY 4.0** (Creative Commons Attribution 4.0 International License).

You are free to:
- **Share**: Copy and redistribute the material
- **Adapt**: Remix, transform, and build upon the material

Under the following terms:
- **Attribution**: You must give appropriate credit, provide a link to the licence, and indicate if changes were made

---

## 🙏 Acknowledgments

- **Claude Sonnet 4.5** (Anthropic): AI assistance in research and implementation
- **Open-source community**: TinyBERT, ONNX Runtime, Tesseract.js contributors
- **Medical community**: Clinical guidance on biomarker plausibility

---

**Last Updated**: December 14, 2025
