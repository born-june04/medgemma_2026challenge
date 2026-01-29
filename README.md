# Healthcare Demo Pipeline (Audio + CXR + Evidence + Report)

This is a **minimal, interpretable demo pipeline** for a clinician-facing support tool. It is **not** a diagnostic system.

Goals:
- Provide **audio-based evidence** (classification + hierarchical physiological analysis)
- Provide **CXR visual evidence** via occlusion-based attribution + hierarchical radiological analysis
- Provide an optional **LLM-generated draft report** (MedGemma) from the collected multimodal evidence

## Repo structure (current)
```
Agent1_Audio/
├── encoders/             # HeAR audio encoder
├── classifiers/          # Audio classification head
└── physiology/           # Hierarchical audio physiology analysis (NEW)
    ├── features.py       # Audio feature extraction
    ├── analyzer.py       # 3-level hierarchical reasoning
    └── plan.md           # Design document

Agent2_Image/
├── encoders/             # MedSigLIP image encoder
├── classifiers/          # Image classification head
├── utils/                # Occlusion-based attribution
└── physiology/           # Hierarchical CXR radiological analysis (NEW)
    ├── features.py       # CXR feature extraction (zonal, texture)
    └── analyzer.py       # 3-level hierarchical reasoning

pipeline/                 # Pipeline core + report adapter
configs/                  # config.yaml with physiology settings
data/                     # pairs.csv + datasets
outputs/
├── evidence/
│   ├── physiology/       # Audio hierarchical analysis
│   └── cxr_physiology/   # CXR hierarchical analysis
├── gradcam/              # Visual attribution overlays
└── reports/              # MedGemma-generated reports

physiology_explain/       # Testing & validation
├── DOCUMENTATION.md          # Complete hierarchical analysis guide
├── test_audio_analyzer.py    # Audio physiology tests
└── test_cxr_analyzer.py      # CXR physiology tests

pipeline.py               # CLI entrypoint
run_pipeline_with_physiology.sh  # Quick start script
README.md
```

---

## 🚀 Quick Start

### Option 1: Use the shell script (easiest)
```bash
ssh g3099  # GPU node
cd /gscratch/scrubbed/june0604/medgemma_2026challenge
./run_pipeline_with_physiology.sh
```

### Option 2: Run manually
```bash
conda activate kaggle
python pipeline.py \
  --config configs/config.yaml \
  --patient-id P0001 \
  --pairs-index data/pairs.csv
```

### Option 3: Direct paths (no CSV)
```bash
python pipeline.py \
  --config configs/config.yaml \
  --audio /path/to/audio.png \
  --image /path/to/cxr.jpeg
```

---

## 🔬 Hierarchical Physiological Analysis (NEW!)

This pipeline now includes **3-level hierarchical reasoning** for both audio and CXR data, mimicking clinical decision-making.

**📖 For complete documentation, see [`physiology_explain/DOCUMENTATION.md`](physiology_explain/DOCUMENTATION.md)**

Quick overview:

### Audio Physiology Analysis

**Level 1: Broad Pathophysiology**
- Cluster A: Infectious/Inflammatory (COVID-19, Pneumonia, TB)
- Cluster B: Structural/Pleural (Pneumothorax, Atelectasis)
- Cluster C: Mass/Fluid (Cancer, Edema, Consolidation)

**Level 2: Pattern Recognition**
- Dry vs Wet cough signature
- Silent chest vs Diminished ventilation
- Fine crackles vs Dense consolidation vs Monophonic wheeze

**Level 3: Disease-Specific Biomarkers**
- COVID-19: High-harmonic dry cough (spectral centroid >2000Hz)
- Pneumonia: Low-freq mucus resonance (<1000Hz)
- TB: Chronic bouts (temporal burstiness >0.7)
- Pneumothorax: "Silent chest" (high-freq energy <0.05)
- Edema: Fine crackles (high-freq energy >0.3)
- Cancer: Monophonic wheeze (narrow bandwidth <600Hz)

### CXR Physiology Analysis

**Level 1: Broad Radiological Category**
- Cluster A: Increased Opacity (COVID-19, TB, Pneumonia, Edema, Consolidation)
- Cluster B: Structural Changes (Pneumothorax, Atelectasis, Cancer)

**Level 2: Distribution & Texture Patterns**
- Peripheral vs Central vs Apical distribution
- Ground Glass Opacity (GGO) vs Dense Consolidation vs Hyperlucency

**Level 3: Disease-Specific Visual Biomarkers**
- COVID-19: Peripheral predominance (ratio >1.3) + GGO (entropy >0.65)
- Pneumonia: Asymmetric lobar consolidation (homogeneity >0.45)
- TB: Apical predominance (ratio >1.4) + upper lobe concentration
- Pneumothorax: Hyperlucency (opacity <0.30) + unilateral
- Edema: Central bat-wing pattern (peripheral ratio <0.7)
- Cancer: Focal unilateral mass with heterogeneous texture

---

## 📊 Output Structure

### 1. Audio Hierarchical Analysis
`outputs/evidence/physiology/<patient_id>/hierarchical_analysis.json`

```json
{
  "hierarchical_analysis": {
    "level_1": {
      "category": "Cluster A: Infectious/Inflammatory",
      "confidence": 0.92,
      "evidence": ["High cough rate: 7.5/min", ...]
    },
    "level_2": {
      "pattern_type": "Dry cough signature (viral-like)",
      "features": {...}
    },
    "level_3": {
      "primary_candidate": "1. COVID-19",
      "primary_score": 0.85,
      "evidence_for_primary": [
        "High-harmonic dry signature: spectral centroid 2300 Hz",
        "Energy concentrated in higher frequencies (ratio: 0.350)"
      ],
      "alternative_candidate": "8. Pneumonia",
      "evidence_against_alternative": [...]
    },
    "physiological_conclusion": "Audio analysis indicates..."
  },
  "raw_features": {...}
}
```

### 2. CXR Hierarchical Analysis
`outputs/evidence/cxr_physiology/<patient_id>/hierarchical_analysis.json`

```json
{
  "hierarchical_analysis": {
    "level_1": {
      "category": "Cluster A: Increased Opacity",
      "confidence": 1.0,
      "evidence": ["Increased opacity: score 0.563", ...]
    },
    "level_2": {
      "pattern_type": "Peripheral Distribution with GGO",
      "features": {
        "distribution": "Peripheral predominance (ratio: 1.45)",
        "texture": "High entropy (0.680) - hazy pattern"
      }
    },
    "level_3": {
      "primary_candidate": "1. COVID-19",
      "evidence_for_primary": [
        "Peripheral zone predominance (ratio: 1.45 >1.3)",
        "Ground glass opacity pattern",
        "Bilateral involvement (symmetry: 0.920)"
      ]
    },
    "radiological_conclusion": "CXR analysis indicates..."
  },
  "raw_features": {...}
}
```

### 3. Visual Evidence (Occlusion Attribution)
`outputs/gradcam/<patient_id>/<pred_label>/`
- `original.png` - Original CXR image
- `occlusion_decrease_overlay.png` - Evidence-for regions (red = strongest)
- `occlusion_increase_overlay.png` - Distractor regions

### 4. MedGemma Report
`outputs/reports/<patient_id>/report.txt`

The LLM-generated report now includes both audio and CXR hierarchical analyses with biological reasoning.

---

## 🧪 Testing the Analyzers

Test the hierarchical analyzers independently:

```bash
cd physiology_explain

# Test audio analyzer with synthetic patterns
python test_audio_analyzer.py

# Test audio analyzer with real audio file
python test_audio_analyzer.py /path/to/audio.png

# Test CXR analyzer with synthetic patterns
python test_cxr_analyzer.py

# Test CXR analyzer with real CXR image
python test_cxr_analyzer.py /path/to/cxr.jpeg
```

Each test includes 6 synthetic test cases covering all major disease patterns.

---

## ⚙️ Configuration

Enable/disable physiological analysis in `configs/config.yaml`:

```yaml
# Audio physiology analysis
physiology:
  enabled: True
  output_dir: "outputs/evidence/physiology"

# CXR physiology analysis
cxr_physiology:
  enabled: True
  output_dir: "outputs/evidence/cxr_physiology"

# Visual attribution
gradcam:
  enabled: True
  targets: ["occlusion"]
  output_dir: "outputs/gradcam"
  alpha: 0.3

# MedGemma report generation
report:
  enabled: True
  model_id: "google/medgemma-1.5-4b-it"
  output_dir: "outputs/reports"
  max_new_tokens: 512
  temperature: 0.2
```

---

## 📖 Documentation

- **Complete Guide**: [`physiology_explain/DOCUMENTATION.md`](physiology_explain/DOCUMENTATION.md) - Comprehensive documentation for hierarchical analysis
  - System architecture and 3-level reasoning
  - Audio physiological analysis (all 9 diseases)
  - CXR radiological analysis (all 9 diseases)
  - Implementation details and thresholds
  - Example reports with multimodal integration
  - Testing and validation procedures
- **Design Document**: `Hierarchical_Physio_Model_Design_Doc.md.pdf` - Clinical rationale
- **Audio Plan**: `Agent1_Audio/physiology/plan.md` - Audio analysis design

---

## 🎯 Key Features

### Multimodal Evidence Integration
- **Audio analysis** detects acoustic signatures (dry vs wet cough, silent chest, wheezes)
- **CXR analysis** detects visual patterns (peripheral GGO, apical cavities, consolidation)
- **Cross-modal validation** identifies agreement/disagreement between modalities
- **Biological story-telling** provides quantitative evidence with clinical reasoning

### Interpretability
- All thresholds and decisions are explicit and traceable
- Each conclusion is backed by specific measurements (e.g., "spectral centroid 2300 Hz")
- Differential diagnosis shows why alternatives were ruled out
- Natural language conclusions explain the reasoning process

### Clinical Safety
- **Not a diagnostic tool** - outputs are hypothesis-level evidence only
- Designed for clinician review, not autonomous decision-making
- Includes caveats and recommendations for additional testing
- Transparent about confidence levels and uncertainties

---

## 🔧 Checkpoints and Encoders

By default, the pipeline uses **stub encoders** with the same interfaces as the real models.

To use real encoders, set paths in `configs/config.yaml`:
```yaml
audio_checkpoint: /path/to/hear_checkpoint.pt
image_checkpoint: /path/to/cxr_checkpoint.pt
use_stub_encoders: false
```

---

## ⚠️ Clinical Limitations and Safety Notes

- **Not a diagnostic tool**. Outputs are **signals** and **hypotheses**, not diagnoses.
- Scores are **heuristic** and not calibrated to clinical risk.
- Physiological proxies are **conceptual** and meant for **interpretability**, not validation.
- Always interpret results alongside clinical context and standard of care.
- The hierarchical analysis is designed to support, not replace, clinical judgment.
- All biomarkers and thresholds are based on literature review and require validation.

---

## 📚 References

This implementation is inspired by clinical decision-making processes and incorporates:
- Acoustic analysis principles from pulmonary auscultation literature
- Radiological patterns from CXR interpretation guidelines
- Hierarchical reasoning frameworks from diagnostic reasoning research

For research and educational purposes only.
