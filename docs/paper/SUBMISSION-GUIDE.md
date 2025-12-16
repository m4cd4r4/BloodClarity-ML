# Paper Submission Guide

This directory contains three versions of the BloodVital-ML research paper for different submission venues.

## 📄 Available Formats

### 1. LaTeX Version (Recommended for Journals)
**Files:**
- `paper.tex` - Main LaTeX document
- `references.bib` - BibTeX references (Vancouver style)

**Best For:**
- JAMIA (Journal of the American Medical Informatics Association)
- JBI (Journal of Biomedical Informatics)
- AMIA Annual Symposium
- ICML, NeurIPS (Machine Learning conferences)

**How to Compile:**

```bash
# Method 1: Using pdflatex (recommended)
cd docs/paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex  # Run twice for references

# Output: paper.pdf

# Method 2: Using Overleaf (online)
# 1. Go to https://www.overleaf.com
# 2. New Project → Upload Project
# 3. Upload paper.tex and references.bib
# 4. Click "Recompile"
# 5. Download PDF
```

**Requirements:**
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Packages: natbib, hyperref, amsmath, booktabs, graphicx

---

### 2. Word-Compatible Version
**File:** `paper-word-version.md`

**Best For:**
- Medical journals preferring Word submissions
- Quick edits and collaborative review
- Journals without LaTeX support

**How to Convert to Word:**

```bash
# Method 1: Open directly in Microsoft Word
# 1. Right-click paper-word-version.md
# 2. Open With → Microsoft Word
# 3. File → Save As → Word Document (.docx)

# Method 2: Using Pandoc (command-line)
pandoc paper-word-version.md -o paper.docx \
  --reference-doc=custom-reference.docx \
  --citeproc

# Method 3: Copy-paste into Word
# 1. Open paper-word-version.md in text editor
# 2. Copy all content
# 3. Paste into Word
# 4. Apply formatting (headings, tables, etc.)
```

**How to Convert to PDF from Word:**
1. Open `paper-word-version.md` in Microsoft Word
2. File → Export → Create PDF/XPS
3. Save as `paper.pdf`

---

### 3. bioRxiv/medRxiv Preprint Submission

**bioRxiv** is for biology/computer science papers
**medRxiv** is specifically for medical research

#### Step-by-Step Submission to bioRxiv:

**1. Prepare PDF:**

Option A - From LaTeX:
```bash
cd docs/paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
# Creates paper.pdf
```

Option B - From HTML (current paper):
```bash
# Open docs/paper/index.html in Chrome
# Press Ctrl+P (Print)
# Select "Save as PDF"
# Save as paper.pdf
```

**2. Go to bioRxiv:**
- Visit: https://submit.biorxiv.org
- Create account / Login

**3. New Submission:**
- Click "New Submission"
- Accept terms and conditions

**4. Upload Manuscript:**
- Upload `paper.pdf`
- bioRxiv will process and generate preview

**5. Manuscript Information:**

**Title:**
```
Clinical-Grade Browser-Based Machine Learning for Medical Biomarker Extraction from Laboratory Reports
```

**Authors:**
```
BloodVital Research Team
```

**Corresponding Author:**
```
Email: research@bloodvital.com
```

**Abstract:**
(Copy from paper - max 300 words)

**Subject Areas** (select relevant):
- ✅ Bioinformatics
- ✅ Medical Informatics
- ✅ Computer Science - Artificial Intelligence
- ✅ Computer Science - Machine Learning

**Keywords:**
```
medical informatics, named entity recognition, browser-based machine learning, clinical NLP, privacy-preserving AI, HIPAA compliance, biomarker extraction, laboratory reports
```

**6. Licence:**
- Select: **CC BY 4.0** (allows free sharing with attribution)

**7. Conflicts of Interest:**
```
None declared. This is independent research with no commercial conflicts.
```

**8. Funding Statement:**
```
This research was self-funded with no external grants or institutional support.
```

**9. Author Contributions:**
```
The BloodVital Research Team conceived the study, developed the methodology, implemented the system, conducted experiments, analysed results, and wrote the manuscript.
```

**10. Review and Submit:**
- Preview PDF rendering
- Check metadata
- Submit for review
- bioRxiv moderators will review (1-2 days)

**11. After Acceptance:**
- bioRxiv assigns DOI (e.g., https://doi.org/10.1101/2025.12.14.XXXXXX)
- Paper becomes publicly accessible
- Can update with new versions if needed

---

## 📊 Journal-Specific Formatting

### JAMIA (Journal of the American Medical Informatics Association)

**Requirements:**
- Format: LaTeX or Word
- Abstract: Max 250 words, structured (Background, Objective, Methods, Results, Conclusions)
- References: Vancouver style (numbered)
- Figures: Separate files, 300+ DPI
- Tables: In-text or separate

**Submission Portal:** https://academic.oup.com/jamia

**Use:** `paper.tex` (already formatted for JAMIA)

**Changes Needed:**
- Add author affiliations
- Add author ORCID IDs
- Prepare figures as separate .tif or .eps files
- Add cover letter

---

### BMC Medical Informatics and Decision Making

**Requirements:**
- Format: LaTeX, Word, or PDF
- Abstract: Max 350 words (Background, Methods, Results, Conclusions)
- References: Vancouver style
- Open Access: $2,490 APC (Article Processing Charge)

**Submission Portal:** https://bmcmedinformdecismak.biomedcentral.com/submission-guidelines

**Use:** `paper.tex` or `paper-word-version.md`

---

### Journal of Biomedical Informatics (JBI)

**Requirements:**
- Format: Word or LaTeX
- Abstract: Max 200 words
- References: Vancouver or Harvard style
- Figures: Separate files, high resolution

**Submission Portal:** https://www.editorialmanager.com/yjbin/

**Use:** `paper.tex`

---

### ICML / NeurIPS (Machine Learning Conferences)

**Requirements:**
- Format: LaTeX only (specific template required)
- Page Limit: 8 pages (+ unlimited references/appendices)
- Anonymous Submission: Remove author info for review

**Download ICML Template:** https://icml.cc/Conferences/2025/StyleAuthorInstructions

**Modifications Needed:**
1. Download ICML LaTeX template
2. Copy content from `paper.tex` into template
3. Remove author names/affiliations (double-blind review)
4. Compress to fit 8-page limit (move details to appendix)

---

## 🔧 Compilation Troubleshooting

### LaTeX Errors

**Error: "File `natbib.sty' not found"**
```bash
# Install missing packages
# Ubuntu/Debian:
sudo apt-get install texlive-latex-extra

# macOS (MacTeX):
sudo tlmgr install natbib

# Windows (MiKTeX):
# MiKTeX Package Manager → Search "natbib" → Install
```

**Error: "Undefined citations"**
```bash
# Run bibtex to process references:
bibtex paper
pdflatex paper.tex  # Run again after bibtex
```

**Error: "Overfull \hbox" warnings**
```
# Not critical - just formatting warnings
# Can ignore for preprint submission
```

---

### Word Conversion Issues

**Markdown not rendering properly:**
- Use Pandoc for better conversion:
  ```bash
  pandoc paper-word-version.md -o paper.docx
  ```

**Tables not formatting:**
- Manually adjust table alignment in Word
- Or use "Convert Text to Table" feature

**Equations not rendering:**
- Replace LaTeX equations with Word's Equation Editor
- Or submit as PDF instead

---

## 📋 Pre-Submission Checklist

Before submitting to any venue:

### Content Checks
- [ ] All author names and affiliations correct
- [ ] Abstract within word limit
- [ ] All figures cited in text
- [ ] All tables cited in text
- [ ] All references cited in text
- [ ] No "TODO" or placeholder text
- [ ] Acknowledgements included (if applicable)
- [ ] Conflicts of interest statement
- [ ] Data availability statement

### Format Checks
- [ ] Page numbers present
- [ ] Line numbers (if journal requires)
- [ ] Figures at correct resolution (300+ DPI)
- [ ] Tables formatted consistently
- [ ] References in correct style (Vancouver/Harvard)
- [ ] PDF renders correctly (no missing fonts, figures)

### Ethical Checks
- [ ] No patient-identifying information
- [ ] Ethics approval (if human data used) - **N/A** (synthetic data only)
- [ ] Data sharing statement - **Our synthetic data is publicly available**
- [ ] Code availability - **GitHub: bloodvital-ml**

---

## 🎯 Recommended Submission Strategy

**Phase 1: Preprint (Immediate)**
1. Submit to **bioRxiv** (fast, free, immediate visibility)
2. Share DOI on Twitter, LinkedIn, ResearchGate
3. Get early feedback from community

**Phase 2: Journal (1-2 weeks after preprint)**
1. Target: **JAMIA** (high-impact medical informatics)
2. Expect 4-8 week review time
3. Respond to reviewer comments

**Phase 3: Conference (If rejected)**
1. Backup: **AMIA Annual Symposium** (next deadline)
2. Or: **ICML Workshop on ML for Health**

---

## 📧 Sample Cover Letter (JAMIA Submission)

```
Dear Editor-in-Chief,

We are pleased to submit our manuscript titled "Clinical-Grade Browser-Based Machine Learning for Medical Biomarker Extraction from Laboratory Reports" for consideration as an original research article in the Journal of the American Medical Informatics Association.

This work addresses a critical challenge in medical informatics: achieving clinical-grade accuracy (≥98%) for biomarker extraction in browser-deployable models with complete offline capability. Our five-component system achieves 98.8% accuracy while maintaining privacy compliance (HIPAA/GDPR) and eliminating cloud costs ($162,000 annual savings).

The key innovation is demonstrating that domain-specific optimisation enables clinical-grade NER in browser environments—overcoming the perceived trade-off between model accuracy and deployment constraints. This has significant implications for privacy-preserving medical AI and democratising access to automated clinical document processing.

This manuscript has not been published elsewhere and is not under consideration by any other journal. All authors have approved the manuscript and agree with its submission to JAMIA.

We suggest the following reviewers:
1. [Name], [Institution], [Email] - Expert in medical NER
2. [Name], [Institution], [Email] - Expert in browser-based ML
3. [Name], [Institution], [Email] - Expert in clinical informatics

Thank you for considering our manuscript.

Sincerely,
BloodVital Research Team
Correspondence: research@bloodvital.com
```

---

## 📞 Support

For questions about paper formats or submission:
- **Email**: research@bloodvital.com
- **GitHub**: https://github.com/yourusername/bloodvital-ml
- **Documentation**: See README.md in paper directory

---

**Last Updated**: December 14, 2025
**Version**: 1.0
