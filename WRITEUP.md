# Multimodal Pulmonary Diagnostic Assistant

**Team:** June Lee, Jake Chansky \
**Competition:** [MedGemma Impact Challenge 2026](https://www.kaggle.com/competitions/med-gemma-impact-challenge) \
**Repository:** [GitHub](https://github.com/born-june04/medgemma_2026challenge)

### Team & Roles

| Member | Role | Specialty |
|---|---|---|
| **June Lee** (Lead) | Pipeline Architecture, Physiological Reasoning Model | Designed the 3-level hierarchical reasoning framework, built the multimodal fusion pipeline, and developed the Audio/CT physiology analyzers |
| **Jake Chansky** | Agent Workflow, Data Engineering | Orchestrated the agentic workflow deployment, curated and structured the multi-class Chest Diseases Dataset, built evaluation infrastructure |

---

## Problem Statement

Pulmonary diseases remain a leading cause of morbidity and mortality worldwide, disproportionately affecting low-resource settings where specialist radiologists and pulmonologists are scarce. Current AI diagnostic tools often function as opaque classifiers — outputting a disease label and confidence score without explaining the underlying clinical reasoning. This opacity undermines clinician trust and limits the educational value of AI-assisted diagnosis.

**Unmet need:** Clinicians need AI tools that provide *interpretable, evidence-based reasoning* — not just predictions, but transparent explanations that map to established clinical decision-making frameworks. This is especially critical in settings where the AI output may be the primary decision-support resource.

---

## Solution Overview

We present a **Multimodal Pulmonary Diagnostic Assistant** — an agentic pipeline that analyzes three complementary data modalities (cough/breath audio, chest X-rays, and CT scans) through specialized AI agents, producing hierarchical clinical evidence and a synthesized report.

### Key Innovation: 3-Level Hierarchical Clinical Reasoning

Rather than reducing diagnosis to a single classifier output, our system mimics the clinical reasoning process:

1. **Level 1 — Broad Category:** Classifies the presentation into pathophysiological clusters (e.g., Infectious vs. Structural vs. Mass/Fluid)
2. **Level 2 — Pattern Recognition:** Identifies specific acoustic or radiological patterns (e.g., "Dry cough signature" or "Peripheral GGO distribution")
3. **Level 3 — Differential Diagnosis:** Scores individual diseases against quantitative biomarkers with explicit thresholds, providing evidence *for* the primary candidate and evidence *against* alternatives

This hierarchical approach produces a **biological story** — a chain of reasoning that clinicians can follow, verify, and learn from.

---

## Technical Approach

### HAI-DEF Model Integration

| Model | Modality | Role |
|---|---|---|
| **HeAR** (`google/hear-pytorch`) | Audio | Frozen encoder producing embeddings from cough/breath spectrograms |
| **MedSigLIP-448** (`google/medsiglip-448`) | CXR | Frozen encoder producing embeddings from chest X-ray images |
| **MedSigLIP-448** (`google/medsiglip-448`) | CT | Frozen encoder producing embeddings from CT scan slices |
| **MedGemma 1.5-4B-IT** (`google/medgemma-1.5-4b-it`) | Text | Generates structured clinical report drafts from multimodal evidence |

### Agentic Architecture

The pipeline operates as three **specialized agents** coordinated by a fusion layer:

- **Agent 1 (Audio):** HeAR encoder → MLP classifier head → Hierarchical physiological analyzer. Extracts acoustic biomarkers (spectral centroid, HF energy ratio, temporal burstiness, cough rate) and maps them through a 3-level clinical reasoning tree covering 9 disease classes.

- **Agent 2 (CXR):** MedSigLIP encoder → MLP classifier head → Hierarchical radiological analyzer + Occlusion-based visual attribution. Extracts zonal opacity distribution, texture features, and symmetry metrics, then applies radiological reasoning patterns.

- **Agent 3 (CT):** MedSigLIP encoder → MLP classifier head → Hierarchical parenchymal analyzer. Analyzes axial CT slices for GGO extent, consolidation, crazy-paving patterns, peripheral distribution, and bilaterality — providing higher sensitivity for early viral pneumonitis detection.

- **Evidence Fusion:** 3-way cross-modal agreement check between all agents, followed by MedGemma report synthesis.

### Disease Coverage (9 Classes)

COVID-19, Lung Cancer, Consolidation, Atelectasis, Tuberculosis, Pneumothorax, Edema, Pneumonia, and Normal — each with modality-specific biomarker profiles and explicit decision thresholds derived from clinical literature.

### Prediction Architecture: Embedding → Classifier → Clinical Reasoning

Each agent follows a 3-stage prediction pipeline:

```
Stage 1: Frozen encoder extracts dense embedding (768-dim)
    Audio:  HeAR(spectrogram)     → 768-dim embedding
    CXR:    MedSigLIP(image)      → 768-dim embedding
    CT:     MedSigLIP(image)      → 768-dim embedding

Stage 2: Learned MLP classifier head maps embedding → disease probabilities
    Embedding (768) → Linear(768, 256) → ReLU → Dropout(0.3)
                     → Linear(256, 9)  → Softmax → 9-class probability vector

Stage 3: Hierarchical Physiology Analyzer interprets probabilities + raw features
    (class_probs, extracted_features) → 3-Level Clinical Reasoning Tree
    L1: Pathophysiological category (Infectious / Structural / Mass)
    L2: Specific pattern (Dry cough / GGO / Consolidation)
    L3: Disease-level score with evidence chain
```

The classifier heads are **separately trained per modality** on the 9-class Chest Diseases Dataset (326 files) using the frozen embeddings as input. This two-stage approach (frozen encoder + learned head) enables efficient fine-tuning without modifying the pretrained foundation models.

**Late Fusion:** After individual agent predictions, the image embedding is optionally concatenated with the audio class probabilities (`torch.cat([image_emb, audio_probs])`) to create a fused representation for integrated analysis.

### Disagreement Resolution Protocol

When agents disagree on the primary diagnosis, the system follows a structured 3-step resolution protocol rather than silently defaulting to majority vote:

**Step 1 — DETECT:** Disagreement is flagged when any agent's top-1 prediction differs from another agent's top-1 prediction *and* both have confidence >0.5.

**Step 2 — WEIGHT:** Each modality receives a clinical specificity weight:
- CT: ×1.2 (highest spatial resolution, most diagnostically specific)
- CXR: ×1.0 (standard baseline)
- Audio: ×0.8 (complementary but less diagnostically specific)

**Step 3 — DECIDE:** Weighted confidence scores are summed per disease across all agents. The highest-scoring disease becomes the primary hypothesis, but a **NEEDS CONFIRMATION** flag is raised with specific recommended confirmatory tests.

Critically, the system **preserves all agent opinions** in the output report. Disagreement is treated as clinically informative — it may indicate co-morbidity (e.g., viral pneumonitis with bacterial superinfection) rather than model error. See the [Disagreement Case Report](docs/sample_report/DISAGREEMENT_CASE_REPORT.md) for a worked example.

### Interpretability Features

- **Quantitative evidence:** Every conclusion cites specific measurements (e.g., "spectral centroid: 2340 Hz exceeds 2000 Hz threshold for COVID-19")
- **Differential reasoning:** Explicitly shows why alternatives were ruled out
- **Visual attribution:** Occlusion-based heatmaps highlight which CXR regions influence the prediction
- **Cross-modal validation:** Identifies 3-way agreement/disagreement between audio, CXR, and CT findings

---

## Results & Demonstration

### Example: COVID-19 Case

**Audio Agent** detects high-harmonic dry cough (spectral centroid: 2340 Hz, HF energy ratio: 0.38) → classified as Infectious/Inflammatory → COVID-19 primary candidate (score: 0.89).

**CXR Agent** identifies bilateral peripheral GGO (peripheral ratio: 1.52, texture entropy: 0.72, symmetry: 0.91) → classified as Increased Opacity → COVID-19 primary candidate (score: 0.91).

**CT Agent** detects bilateral subpleural ground glass (GGO extent: 64%, crazy-paving: 41%, peripheral distribution: 78%) → COVID-19 primary candidate (score: 0.94).

**3-way cross-modal agreement:** All three modalities converge on COVID-19 with combined confidence of 94%.

**MedGemma report** synthesizes the full 3-modality evidence package into a structured clinical draft with impression, per-agent evidence summaries, 3-way agreement confirmation, and recommended next steps (RT-PCR confirmation, serial CT imaging).

### Example: Pneumothorax Case

Audio "silent chest" (HF energy: 0.03) directly correlates with CXR hyperlucency (opacity: 0.18), demonstrating how the system captures physiological consistency across modalities — air in the pleural space simultaneously blocks sound transmission and creates radiographic lucency.

> Full sample outputs are available in `docs/sample_outputs/` and can be regenerated locally via `python generate_sample_outputs.py` (no GPU required).

---

## Product Feasibility & Deployment

### Application Stack

| Component | Technology | Rationale |
|---|---|---|
| Audio Encoder | HeAR (frozen, 768-dim) | Lightweight inference, no GPU required for embedding extraction |
| Image Encoder | MedSigLIP-448 (frozen, 768-dim) | Single forward pass per image, compatible with edge GPUs |
| Classifier Heads | 2-layer MLP (768→256→9) | <1ms inference, ~100KB per modality |
| Report Generator | MedGemma 1.5-4B-IT | Quantized 4-bit via GPTQ for edge deployment |
| Physiological Analyzer | Rule-based reasoning tree | Pure Python, zero latency, fully transparent |

### Deployment Strategy

- **Edge-first architecture:** All encoders are frozen and open-weight — no cloud API dependency. A single NVIDIA Jetson Orin or Apple M-series laptop can run the full pipeline.
- **Audio-only screening mode:** In the most resource-constrained settings, Agent 1 (HeAR) runs standalone on a smartphone microphone, providing initial screening without any imaging equipment.
- **Modular scale-up:** Start with audio-only, add CXR (portable X-ray), then CT when available — the pipeline gracefully handles 1, 2, or 3 modalities.
- **EHR integration:** Structured JSON outputs (`docs/sample_outputs/`) are designed for direct integration into FHIR-compatible electronic health record systems.

### Inference Pipeline

```
Patient data → [Parallel agent inference ~2s] → Evidence fusion → MedGemma report draft → Clinician review
```

Total end-to-end latency: **~5 seconds** on consumer hardware (M2 MacBook Pro) for 3-modality analysis including report generation.

---

## Impact & Vision

- **Global health equity:** 4.7 billion people lack access to diagnostic imaging (WHO, 2023). Our audio-first screening pipeline requires only a smartphone, enabling point-of-care respiratory assessment in community health settings.
- **Clinician force multiplier:** In sub-Saharan Africa, there is 1 radiologist per 1 million people. AI-generated evidence chains allow general practitioners to make specialist-level pulmonary assessments with transparent supporting evidence.
- **Clinician education:** The 3-level hierarchical reasoning framework serves as a teaching tool — showing *why* specific radiological or acoustic patterns map to specific diseases, not just *that* they do.
- **Privacy-preserving:** All processing happens locally. No patient data leaves the device — critical for regulatory compliance (HIPAA, GDPR) and trust in healthcare AI.

---

## Agentic Workflow Design

Our pipeline is an inherently **agentic system** where three specialized AI agents independently analyze complementary data modalities, then coordinate through a structured evidence fusion protocol:

1. **Autonomous analysis:** Each agent independently processes its modality through a frozen encoder → classifier → physiological reasoning pipeline, producing structured evidence without knowledge of other agents' findings.
2. **Cross-modal validation:** The fusion layer checks whether agents agree or disagree, and quantifies the strength of agreement.
3. **Structured disagreement handling:** When agents disagree, the system doesn't silently pick a winner — it raises a **NEEDS CONFIRMATION** flag with weighted voting resolution and specific recommended confirmatory tests (see [Disagreement Case Report](docs/sample_report/DISAGREEMENT_CASE_REPORT.md)).
4. **Report synthesis:** MedGemma synthesizes all agent evidence into a clinician-facing report, preserving all individual agent perspectives.

This agentic design mirrors how clinical teams operate: multiple specialists independently evaluate a case, then convene to discuss and reconcile their findings.

---

## Limitations & Safety

- **Not a diagnostic tool** — all outputs are hypothesis-level evidence requiring clinical verification
- **Biomarker thresholds** are derived from published literature but have not been validated in a prospective clinical trial
- **Dataset limitations** — the Chest Diseases Dataset (326 files, 9 classes) is small; performance at scale requires larger validated datasets
- **Audio analysis** relies on spectrogram representations rather than raw waveforms, which may lose temporal fine-structure
- **Model calibration** — classifier confidence scores are not calibrated to clinical risk and should not be interpreted as diagnostic probabilities
