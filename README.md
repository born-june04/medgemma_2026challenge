# Healthcare Demo Pipeline (Audio + CXR + Evidence + Report)

This is a **minimal, interpretable demo pipeline** for a clinician-facing support tool. It is **not** a diagnostic system.

Goals:
- Provide **audio-based evidence** (classification + interpretable physiology features)
- Provide **CXR visual evidence** via occlusion-based attribution (spatial “what influenced the score”)
- Provide an optional **LLM-generated draft report** (MedGemma) from the collected evidence

## Repo structure (current)
```
Agent1_Audio/          # audio encoder + audio classifier training code + physiology features
Agent2_Image/          # MedSigLIP encoder + CXR classifier training code + occlusion utilities
pipeline/              # pipeline core + report adapter
configs/
data/                  # pairs.csv (patient_id,audio_path,image_path) + datasets (optional)
outputs/               # gradcam/ , evidence/ , reports/
pipeline.py            # CLI entrypoint for the pipeline
README.md
```

## 1) Run the pipeline (recommended starting point)

### A) Run by patient_id (paired index CSV)
You need a CSV with columns: `patient_id,audio_path,image_path` (see `data/pairs.csv`).

```bash
conda activate kaggle
python /gscratch/scrubbed/june0604/medgemma_2026challenge/pipeline.py \
  --config /gscratch/scrubbed/june0604/medgemma_2026challenge/configs/config.yaml \
  --patient-id P0001 \
  --pairs-index /gscratch/scrubbed/june0604/medgemma_2026challenge/data/pairs.csv
```

### B) Run by direct paths
```bash
conda activate kaggle
python /gscratch/scrubbed/june0604/medgemma_2026challenge/pipeline.py \
  --config /gscratch/scrubbed/june0604/medgemma_2026challenge/configs/config.yaml \
  --audio /abs/path/to/audio_or_scalogram.png \
  --image /abs/path/to/cxr.png
```

## 2) Outputs: visual + biological evidence

### A) CXR visual evidence (occlusion overlays)
Controlled by `configs/config.yaml -> gradcam`.

When enabled with `targets: ["occlusion"]`, the pipeline saves:
- `outputs/gradcam/<patient_id>/<pred_label>/original.png`
- `outputs/gradcam/<patient_id>/<pred_label>/occlusion_decrease_overlay.png` (occlusion decreases score = evidence-for)
- `outputs/gradcam/<patient_id>/<pred_label>/occlusion_increase_overlay.png` (occlusion increases score = distractor)

Each occlusion overlay contains a **right-side colorbar**:
- **red ~ 1.0**: strongest effect
- **blue ~ 0.0**: weakest effect

### B) Physiology evidence (interpretable audio features)
Controlled by `configs/config.yaml -> physiology`.

When enabled, the pipeline saves:
- `outputs/evidence/physiology/<patient_id>/physiology.json`

These are simple interpretable features (e.g., cough rate, spectral centroid/bandwidth) intended for **hypothesis-level evidence**, not diagnosis.

## 3) Generate a MedGemma report (optional)
Controlled by `configs/config.yaml -> report`.

Example config:
```yaml
report:
  enabled: True
  model_id: "google/medgemma-4b-it"
  output_dir: "outputs/reports"
  max_new_tokens: 512
  temperature: 0.2
```

When enabled, the pipeline saves:
- `outputs/reports/<patient_id>/prompt.txt`
- `outputs/reports/<patient_id>/inputs.json`
- `outputs/reports/<patient_id>/report.txt` (if generation succeeds)
- `outputs/reports/<patient_id>/error.txt` (if generation fails)

Notes:
- Some models are gated; `configs/config.yaml -> hf_token` is used for automatic login.
- This report is a **draft** based on provided evidence only. Do not treat as clinical truth.

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

## Notes on LLM usage
MedGemma report generation is implemented in `pipeline/reporting/medgemma_report.py` and is fully optional.

## Clinical limitations and safety notes
- **Not a diagnostic tool**. Outputs are **signals** and **hypotheses**, not diagnoses.
- Scores are **heuristic** and not calibrated to clinical risk.
- Physiological proxies are **conceptual** and meant for **interpretability**, not validation.
- Always interpret results alongside clinical context and standard of care.

