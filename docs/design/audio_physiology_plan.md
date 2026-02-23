# Hierarchical Physiological Analysis Model

## 1. Overview & Objective

본 모델은 Audio(CSI)와 Image(CXR/CT) 데이터로부터 **계층적 추론(Hierarchical Reasoning)**을 통해 생물학적으로 해석 가능한 evidence를 제공합니다.

**핵심 원칙**: 
- 단순 feature 나열이 아닌, 임상의의 사고 과정을 모방한 **3단계 Decision Tree**
- Level 1 → Level 2 → Level 3 순으로 병태생리학적 카테고리를 좁혀가는 방식

---

## 2. Three-Level Hierarchy

### Level 1: Broad Pathophysiology (대분류)
가장 먼저 환자가 어떤 병태생리학적 카테고리에 속하는지 구분

| Cluster | Target Diseases | Key Characteristics |
|---------|----------------|---------------------|
| **A: Infectious/Inflammatory** | COVID-19, Pneumonia, Tuberculosis | 염증성 분비물, 기침 패턴 변화 |
| **B: Structural/Pleural** | Pneumothorax, Atelectasis | 호흡음 소실, 폐 용적 변화 |
| **C: Mass/Fluid** | Lung Cancer, Edema, Consolidation | 국소 밀도 증가, 공명음 변화 |

---

## 3. Audio Biomarkers by Disease (Level 2 & 3)

### Cluster A: Infectious/Inflammatory

#### Level 2 (Shared)
- **Spectral Variance**: 불규칙한 신호, 잡음 섞인 호흡
- **High Cough Rate**: 분당 기침 빈도 증가
- **Temporal Burstiness**: 기침의 발작적 패턴

#### Level 3 (Disease-Specific)

**COVID-19 (Viral, Interstitial)**
```
Audio Signature:
- Dry cough pattern: short explosive phase
- High spectral centroid (>2000 Hz) - 마른 기침 특성
- Low energy in 200-500 Hz band (분비물 적음)
- Regular inter-cough intervals (std/mean < 0.5)
```

**Bacterial Pneumonia (Lobar, Alveolar)**
```
Audio Signature:
- Wet/productive cough: longer duration
- High energy in 200-500 Hz (mucus resonance)
- Coarse crackles pattern
- Lower spectral centroid (<1500 Hz)
```

**Tuberculosis (Chronic, Cavitary)**
```
Audio Signature:
- Chronic pattern: high cough rate (>10/min)
- Bursty intervals: high std (temporal_burstiness > 0.7)
- Mixed frequency content (broad spectral bandwidth)
```

---

### Cluster B: Structural/Pleural

#### Level 2 (Shared)
- **Low RMS Energy**: 호흡음 자체가 약함
- **Reduced High-Freq Content**: 환기 감소

#### Level 3 (Disease-Specific)

**Pneumothorax (Air in Pleura)**
```
Audio Signature:
- Severely diminished sound: RMS < 0.2 * baseline
- Absent high-frequency energy (>2000 Hz ratio < 0.05)
- Minimal cough events detected
```

**Atelectasis (Lung Collapse)**
```
Audio Signature:
- Diminished but present sound: RMS 0.3-0.5 * baseline
- Potential fine crackles on re-expansion
- Irregular but present cough pattern
```

---

### Cluster C: Mass/Fluid

#### Level 2 (Shared)
- **Spectral Bandwidth Change**: 공명 특성 변화
- **Moderate Energy Levels**: 완전 소실은 아님

#### Level 3 (Disease-Specific)

**Pulmonary Edema (Heart Failure)**
```
Audio Signature:
- Fine crackles: high-frequency components (>3000 Hz)
- Bilateral pattern: consistent across channels
- Regular but frequent cough (moderate rate)
```

**Lung Cancer (Malignancy)**
```
Audio Signature:
- Localized wheeze: narrow-band peak in spectrum
- Unilateral pattern: asymmetric energy
- Lower cough rate, longer intervals
```

**Consolidation Lung**
```
Audio Signature:
- Dense sound: reduced spectral centroid
- Increased low-frequency energy (200-800 Hz)
- Coarse breath sounds
```

---

## 4. Implementation Algorithm

### Step 1: Extract Audio Features
```python
features = extract_audio_features(audio_path)
# → cough_rate, inter_cough_interval_mean/std, 
#    spectral_centroid, spectral_bandwidth, 
#    high_freq_energy_ratio, temporal_burstiness
```

### Step 2: Level 1 Classification
```python
def classify_level_1(features):
    # Rule-based scoring
    if features.cough_rate > 5 and features.temporal_burstiness > 0.5:
        return "Cluster A: Infectious/Inflammatory"
    elif features.high_freq_energy_ratio < 0.1:
        return "Cluster B: Structural/Pleural"
    else:
        return "Cluster C: Mass/Fluid"
```

### Step 3: Level 2 Pattern Recognition
각 Cluster 내에서 공통 특성 확인

### Step 4: Level 3 Differential Diagnosis
Cluster 내 질환 간 차별화

### Step 5: Generate Evidence JSON
```json
{
  "hierarchical_analysis": {
    "level_1": {
      "category": "Infectious/Inflammatory",
      "confidence": 0.85,
      "evidence": ["High cough rate: 12.5/min", "Temporal burstiness: 0.72"]
    },
    "level_2": {
      "pattern_type": "Inflammatory with secretions",
      "features": {
        "spectral_variance": "High",
        "cough_productivity": "Wet signature detected"
      }
    },
    "level_3": {
      "primary_candidate": "Bacterial Pneumonia",
      "evidence_for": [
        "High energy in 200-500 Hz (mucus resonance)",
        "Low spectral centroid: 1420 Hz",
        "Coarse pattern detected"
      ],
      "alternative_candidate": "COVID-19",
      "evidence_against": [
        "Spectral centroid too low for dry cough",
        "High low-frequency energy inconsistent with viral pattern"
      ],
      "confidence_scores": {
        "Bacterial Pneumonia": 0.78,
        "COVID-19": 0.22
      }
    },
    "physiological_conclusion": "Audio evidence strongly suggests bacterial pneumonia due to wet cough signature with prominent low-frequency mucus resonance, distinguishing it from typical viral (COVID-19) dry cough pattern."
  }
}
```

---

## 5. Key Thresholds (Empirical Guidelines)

| Feature | Normal Range | Abnormal Indicator |
|---------|-------------|-------------------|
| cough_rate_per_min | 0-3 | >5 (infectious), ~0 (structural) |
| spectral_centroid_hz | 1500-2500 | >2500 (dry/COVID), <1500 (wet/pneumonia) |
| high_freq_energy_ratio | 0.15-0.30 | >0.35 (dry), <0.10 (structural loss) |
| temporal_burstiness | 0.2-0.5 | >0.7 (TB/chronic), <0.2 (regular/edema) |
| inter_cough_interval_std | 1-5 sec | >8 (erratic), <0.5 (very regular) |

---

## 6. Clinical Interpretation Notes

**중요**: 이 분석은 **hypothesis-level evidence**이며, 진단이 아닙니다.

- Audio feature만으로는 확정 진단 불가능
- Image evidence와의 cross-validation 필수
- 임상 맥락(증상, 병력, 이학적 검사) 고려 필요
- Story-telling을 위한 biological plausibility 제공이 목적

---

## 7. Integration with Pipeline

```python
# In pipeline/core.py or analysis module
from Agent1_Audio.physiology.analyzer import HierarchicalPhysiologyAnalyzer

analyzer = HierarchicalPhysiologyAnalyzer()
result = analyzer.analyze(audio_features)

# Save to outputs/evidence/physiology/<patient_id>/hierarchical_analysis.json
```
