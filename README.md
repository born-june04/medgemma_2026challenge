# Healthcare Demo App (Audio + CXR)

This is a **minimal, interpretable demo pipeline** for a clinician-facing support tool. It is **not** a diagnostic system.

Goals:
- Provide **signal-level evidence** from cough/breath audio and chest X-ray (CXR)
- Provide **physiological/biomechanical interpretation** of those signals
- Provide a **clinician-facing explanation** with uncertainty and suggested next steps

The demo is designed to run end-to-end **without internet** and **without model checkpoints**, using stubs.

## Repo structure
```
encoders/
signals/
physiology/
llm/
app/
cli.py
configs/
sample_data/
README.md
```

## Quick start (macOS)
1) Create a virtualenv and install requirements:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run the Streamlit app:
```
streamlit run app/streamlit_app.py
```

3) Or run the CLI:
```
python cli.py \
  --audio /path/to/cough.wav \
  --image /path/to/cxr.jpg \
  --intake "Short intake text"
```

## Checkpoints: enable/disable HAI-DEF encoders
By default, the app uses **stub encoders** with the same interfaces as the real models.

To **attempt** real encoders later, set paths in `configs/config.yaml`:
```
audio_checkpoint: /path/to/hear_checkpoint.pt
image_checkpoint: /path/to/cxr_checkpoint.pt
use_stub_encoders: true
```

- If `use_stub_encoders: true`, stubs are always used.
- If `use_stub_encoders: false`, the system will try to load checkpoints. **This repo only includes placeholder code**, so you must implement the real HeAR/CXR load logic.

## Plugging in Med-Gemma later
The LLM is wired via a small adapter in `llm/adapter.py`.

- `llm/placeholder.py` implements a **conservative, rule-based** generator.
- To add Med-Gemma, implement `MedGemmaAdapter.generate()` to return the schema:

```json
{
  "risk_tier": "low | medium | high | insufficient_data",
  "evidence_summary": ["..."],
  "physiological_context": ["..."],
  "recommended_next_actions": ["..."],
  "uncertainties": ["..."]
}
```

Then switch the adapter in `pipeline.py` or add a config flag.

## Clinical limitations and safety notes
- **Not a diagnostic tool**. Outputs are **signals** and **hypotheses**, not diagnoses.
- Scores are **heuristic** and not calibrated to clinical risk.
- Physiological proxies are **conceptual** and meant for **interpretability**, not validation.
- Always interpret results alongside clinical context and standard of care.

## Open-source datasets for testing

### ⚠️ Important: Paired vs. Unpaired Data

**This app processes cough/breath audio and chest X-ray together**, which ideally requires **paired data** (same patient, same timepoint) for end-to-end training and validation. However, **publicly available paired datasets (same patient's audio + X-ray) are extremely rare** due to:

- **Clinical workflow separation**: Audio recording and X-ray imaging are typically separate procedures
- **Privacy/IRB constraints**: Linking audio + imaging increases re-identification risk
- **Data collection complexity**: Requires coordinated protocols and patient consent for both modalities

**Current status**: Most research uses **unpaired datasets** (audio from one source, X-rays from another) and combines them via fusion techniques, though this limits cross-modal correlation learning.

**For this demo**: The pipeline accepts separate audio and X-ray inputs and processes them independently, then combines signals at the interpretation level. This works for demonstration but has limitations for true multimodal learning.

### Cough/Breath Audio Datasets

1. **Coswara Dataset** (IISC Bangalore, India)
   - COVID-19 cough/breath audio dataset
   - GitHub: https://github.com/iiscleap/Coswara-Data
   - Contains cough, breathing, and speech recordings
   - License: Research use

2. **MIT-BIH / PhysioNet**
   - Various respiratory and physiological audio datasets
   - Website: https://physionet.org/
   - Search for "respiratory", "cough", or "breath" datasets
   - Most datasets require data use agreement

3. **AudioSet** (Google Research)
   - Large-scale audio dataset with medical audio categories
   - Website: https://research.google.com/audioset/
   - May contain cough/breath samples in various contexts

### Chest X-Ray Datasets

1. **NIH Chest X-ray 14 Dataset**
   - 14 common thoracic disease labels
   - Download: https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
   - License: CC0 / Public domain
   - Contains ~112K images

2. **MIMIC-CXR Database**
   - Chest X-rays with radiology reports
   - Website: https://physionet.org/content/mimic-cxr/2.0.0/
   - Requires CITI training and data use agreement
   - Contains ~377K images

3. **CheXpert Dataset** (Stanford)
   - Large chest radiograph dataset
   - Website: https://stanfordmlgroup.github.io/competitions/chexpert/
   - Research use license
   - Contains ~224K images

4. **Kaggle Chest X-Ray (Pneumonia) Dataset**
   - Binary classification (Normal vs. Pneumonia)
   - Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
   - Simple structure for quick testing
   - Contains ~5,800 images

### General Healthcare Dataset Repositories

- **Awesome Healthcare Datasets** (GitHub): Comprehensive list of medical datasets
  - https://github.com/geniusrise/awesome-healthcare-datasets
- **serghiou/open-data**: Curated open healthcare databases
  - https://github.com/serghiou/open-data

### Potential Paired/Multimodal Datasets (Limited Availability)

1. **Chest Diseases Using Different Medical Imaging and Cough Sounds** (Mendeley Data)
   - Contains cough sound images (spectrograms), CXR, and CT scans
   - **Note**: Patient-level pairing is unclear - metadata states "meta-information of the patient is not provided"
   - Link: https://data.mendeley.com/datasets/y6dvssx73b/1
   - Use with caution: May not be true patient-level pairs

2. **Research Papers with Multimodal Fusion**
   - Several papers combine Coswara (cough audio) + chest X-ray datasets
   - However, these typically use **separate datasets** combined via fusion, not true patient pairs
   - Example: "Chest X-ray and cough sample based deep learning framework..." (PubMed ID: 36119394)

### Notes on Dataset Usage

- **Data Use Agreements**: Many clinical datasets require IRB approval or data use agreements. Check each dataset's requirements.
- **Licensing**: Verify license terms (research use only, commercial use restrictions, etc.)
- **Privacy**: Ensure datasets are de-identified and comply with HIPAA/GDPR if applicable
- **Paired Data Limitation**: For true end-to-end multimodal learning, consider:
  - **Institutional collaboration**: Partner with hospitals to collect paired data (requires IRB approval)
  - **Proxy pairing**: Match audio and X-ray samples by metadata (age, gender, symptoms) - not ideal but workable for demos
  - **Synthetic pairing**: Use separate datasets but acknowledge limitations in cross-modal correlation
- **Sample Data**: For quick demos, start with NIH Chest X-ray or Kaggle datasets (lower access barriers)

## Sample data
Place sample audio and images in `sample_data/`. This repo does not include medical data.
