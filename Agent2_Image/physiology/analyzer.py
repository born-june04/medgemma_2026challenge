"""Hierarchical Radiological Analysis for CXR Features.

This module implements a 3-level clinical reasoning process for chest X-rays:
- Level 1: Broad radiological category (Increased Opacity / Structural Changes)
- Level 2: Distribution and texture patterns
- Level 3: Disease-specific visual biomarkers

Clinical intent: Provide interpretable, hypothesis-level radiological evidence.
"""

from dataclasses import asdict
from typing import Dict, List, Tuple

from Agent2_Image.physiology.features import CXRFeatures


class HierarchicalCXRAnalyzer:
    """Hierarchical CXR-based radiological analysis engine."""

    # Disease labels matching the dataset
    DISEASES = [
        "1. COVID-19",
        "2. Lungs Cancer",
        "3. Consolidation Lung",
        "4. Atelectasis",
        "5. Tuberculosis",
        "6. Pneumothorax",
        "7. Edema",
        "8. Pneumonia",
        "9. Normal",
    ]

    # Cluster definitions
    CLUSTER_A = ["1. COVID-19", "5. Tuberculosis", "7. Edema", "8. Pneumonia", "3. Consolidation Lung"]
    CLUSTER_B = ["4. Atelectasis", "6. Pneumothorax", "2. Lungs Cancer"]

    def analyze(self, features: CXRFeatures) -> Dict:
        """Perform hierarchical radiological analysis on CXR features.

        Args:
            features: Extracted CXRFeatures object

        Returns:
            Dictionary with hierarchical analysis results
        """
        # Level 1: Classify into broad category
        level_1_result = self._classify_level_1(features)

        # Level 2: Distribution and texture analysis
        level_2_result = self._analyze_level_2(features, level_1_result["category"])

        # Level 3: Disease-specific visual biomarkers
        level_3_result = self._differential_diagnosis(
            features, level_1_result["category"], level_2_result
        )

        # Generate radiological conclusion
        conclusion = self._generate_conclusion(
            level_1_result, level_2_result, level_3_result
        )

        return {
            "hierarchical_analysis": {
                "level_1": level_1_result,
                "level_2": level_2_result,
                "level_3": level_3_result,
                "radiological_conclusion": conclusion,
            },
            "raw_features": asdict(features),
        }

    def _classify_level_1(self, features: CXRFeatures) -> Dict:
        """Level 1: Classify into broad radiological category."""
        scores = {
            "Cluster A: Increased Opacity": 0.0,
            "Cluster B: Structural Changes": 0.0,
            "Normal": 0.0,
        }
        evidence = []

        # Cluster A: Increased opacity (the "whiter" lungs)
        if features.opacity_score > 0.5:
            scores["Cluster A: Increased Opacity"] += 0.5
            evidence.append(
                f"Increased opacity: score {features.opacity_score:.3f} (>0.5 threshold indicates pathological whiteness)"
            )

        if features.mean_intensity > 0.45:
            scores["Cluster A: Increased Opacity"] += 0.3
            evidence.append(
                f"Elevated mean intensity: {features.mean_intensity:.3f} (lung fields appear brighter than normal)"
            )

        # Cluster B: Structural changes (volume loss, lucency)
        if features.opacity_score < 0.35:
            scores["Cluster B: Structural Changes"] += 0.4
            evidence.append(
                f"Low opacity: {features.opacity_score:.3f} (suggests structural issue or hyperlucency)"
            )

        if features.bilateral_symmetry_score < 0.85:
            scores["Cluster B: Structural Changes"] += 0.3
            evidence.append(
                f"Asymmetric pattern: symmetry {features.bilateral_symmetry_score:.3f} (unilateral pathology)"
            )

        # Normal: low opacity, high symmetry, moderate texture
        if 0.30 < features.opacity_score < 0.42 and features.bilateral_symmetry_score > 0.90:
            scores["Normal"] += 0.5
            evidence.append(
                f"Normal appearance: opacity {features.opacity_score:.3f}, symmetry {features.bilateral_symmetry_score:.3f}"
            )

        # Normalize
        total = sum(scores.values())
        if total > 0:
            for key in scores:
                scores[key] /= total

        category = max(scores, key=scores.get)
        confidence = scores[category]

        return {
            "category": category,
            "confidence": float(confidence),
            "evidence": evidence,
            "all_scores": {k: float(v) for k, v in scores.items()},
        }

    def _analyze_level_2(self, features: CXRFeatures, category: str) -> Dict:
        """Level 2: Distribution and texture pattern analysis."""
        if "Cluster A" in category:
            return self._level_2_cluster_a(features)
        elif "Cluster B" in category:
            return self._level_2_cluster_b(features)
        else:  # Normal
            return {
                "pattern_type": "Normal lung fields",
                "features": {
                    "distribution": "Uniform",
                    "texture": "Normal homogeneity"
                }
            }

    def _level_2_cluster_a(self, features: CXRFeatures) -> Dict:
        """Level 2 analysis for Increased Opacity cluster."""
        pattern_features = {}

        # Distribution: Peripheral vs Central
        if features.peripheral_predominance_score > 1.3:
            distribution_pattern = "Peripheral Distribution (COVID-19-like)"
            pattern_features["distribution"] = f"Peripheral predominance (ratio: {features.peripheral_predominance_score:.2f})"
        elif features.peripheral_predominance_score < 0.7:
            distribution_pattern = "Central Distribution (Edema-like)"
            pattern_features["distribution"] = f"Central predominance (ratio: {features.peripheral_predominance_score:.2f})"
        else:
            distribution_pattern = "Focal/Lobar Distribution"
            pattern_features["distribution"] = "Neither central nor peripheral predominance"

        # Apical predominance check (TB)
        if features.apical_predominance_score > 1.4:
            distribution_pattern = "Apical Predominance (TB-like)"
            pattern_features["apical_zone"] = f"Upper lobe concentration (ratio: {features.apical_predominance_score:.2f})"

        # Texture: GGO vs Consolidation
        if features.texture_entropy > 0.65:
            texture_pattern = "Ground Glass Opacity (GGO)"
            pattern_features["texture"] = f"High entropy ({features.texture_entropy:.3f}) - hazy, irregular pattern"
        elif features.texture_homogeneity > 0.45:
            texture_pattern = "Dense Consolidation"
            pattern_features["texture"] = f"High homogeneity ({features.texture_homogeneity:.3f}) - uniform dense opacity"
        else:
            texture_pattern = "Mixed Pattern"
            pattern_features["texture"] = "Intermediate texture characteristics"

        pattern_type = f"{distribution_pattern} with {texture_pattern}"
        return {"pattern_type": pattern_type, "features": pattern_features}

    def _level_2_cluster_b(self, features: CXRFeatures) -> Dict:
        """Level 2 analysis for Structural Changes cluster."""
        pattern_features = {}

        # Volume loss vs Hyperlucency
        if features.bilateral_symmetry_score < 0.85:
            pattern_type = "Asymmetric structural change"
            pattern_features["laterality"] = f"Unilateral (symmetry: {features.bilateral_symmetry_score:.3f})"

            # Check for volume loss (atelectasis)
            if features.opacity_score > 0.35:
                pattern_type = "Volume Loss with Increased Density (Atelectasis-like)"
            else:
                pattern_type = "Hyperlucency (Pneumothorax-like)"
        else:
            pattern_type = "Bilateral structural changes"
            pattern_features["laterality"] = "Symmetric"

        pattern_features["opacity"] = f"{features.opacity_score:.3f}"

        return {"pattern_type": pattern_type, "features": pattern_features}

    def _differential_diagnosis(
        self, features: CXRFeatures, category: str, level_2: Dict
    ) -> Dict:
        """Level 3: Disease-specific differential diagnosis."""
        if "Cluster A" in category:
            return self._diff_cluster_a(features)
        elif "Cluster B" in category:
            return self._diff_cluster_b(features)
        else:
            return {
                "primary_candidate": "9. Normal",
                "primary_score": 0.95,
                "evidence_for_primary": ["Normal radiological appearance"],
                "confidence_scores": {"9. Normal": 0.95}
            }

    def _diff_cluster_a(self, features: CXRFeatures) -> Dict:
        """Differential diagnosis for Increased Opacity cluster."""
        candidates = {}

        # COVID-19: Peripheral GGO, bilateral
        covid_score = 0.0
        covid_evidence_for = []
        covid_evidence_against = []

        if features.peripheral_predominance_score > 1.3:
            covid_score += 0.4
            covid_evidence_for.append(
                f"Peripheral zone predominance (ratio: {features.peripheral_predominance_score:.2f} >1.3) - typical COVID-19 distribution"
            )
        else:
            covid_evidence_against.append(
                f"Lack of peripheral predominance (ratio: {features.peripheral_predominance_score:.2f})"
            )

        if features.texture_entropy > 0.65:
            covid_score += 0.3
            covid_evidence_for.append(
                f"Ground glass opacity pattern (entropy: {features.texture_entropy:.3f}) - hazy appearance"
            )

        if features.bilateral_symmetry_score > 0.85:
            covid_score += 0.2
            covid_evidence_for.append(
                f"Bilateral involvement (symmetry: {features.bilateral_symmetry_score:.3f})"
            )

        candidates["1. COVID-19"] = {
            "score": covid_score,
            "evidence_for": covid_evidence_for,
            "evidence_against": covid_evidence_against,
        }

        # Bacterial Pneumonia: Lobar consolidation, asymmetric, air bronchogram
        pneumonia_score = 0.0
        pneumonia_evidence_for = []
        pneumonia_evidence_against = []

        if features.bilateral_symmetry_score < 0.85:
            pneumonia_score += 0.4
            pneumonia_evidence_for.append(
                f"Asymmetric distribution (symmetry: {features.bilateral_symmetry_score:.3f}) - typical lobar pattern"
            )

        if features.texture_homogeneity > 0.45:
            pneumonia_score += 0.4
            pneumonia_evidence_for.append(
                f"Dense consolidation (homogeneity: {features.texture_homogeneity:.3f}) - air bronchogram likely"
            )
        else:
            pneumonia_evidence_against.append("Texture not sufficiently homogeneous for consolidation")

        if 0.8 < features.peripheral_predominance_score < 1.2:
            pneumonia_score += 0.2
            pneumonia_evidence_for.append("Neither central nor peripheral - focal lobar pattern")

        candidates["8. Pneumonia"] = {
            "score": pneumonia_score,
            "evidence_for": pneumonia_evidence_for,
            "evidence_against": pneumonia_evidence_against,
        }

        # Tuberculosis: Apical predominance, cavitation
        tb_score = 0.0
        tb_evidence_for = []
        tb_evidence_against = []

        if features.apical_predominance_score > 1.4:
            tb_score += 0.6
            tb_evidence_for.append(
                f"Apical zone dominance (ratio: {features.apical_predominance_score:.2f} >1.4) - classic TB pattern"
            )
            tb_evidence_for.append("Upper lobe preference strongly suggests chronic infection")
        else:
            tb_evidence_against.append(
                f"No apical predominance (ratio: {features.apical_predominance_score:.2f})"
            )

        if features.texture_entropy > 0.60:
            tb_score += 0.2
            tb_evidence_for.append("Heterogeneous texture - possible cavitation or mixed pathology")

        if features.bilateral_symmetry_score < 0.85:
            tb_score += 0.2
            tb_evidence_for.append("Asymmetric - consistent with unilateral TB")

        candidates["5. Tuberculosis"] = {
            "score": tb_score,
            "evidence_for": tb_evidence_for,
            "evidence_against": tb_evidence_against,
        }

        # Edema: Central/perihilar, bat-wing, bilateral
        edema_score = 0.0
        edema_evidence_for = []
        edema_evidence_against = []

        if features.peripheral_predominance_score < 0.7:
            edema_score += 0.5
            edema_evidence_for.append(
                f"Central/perihilar predominance (ratio: {features.peripheral_predominance_score:.2f} <0.7) - bat-wing pattern"
            )
        else:
            edema_evidence_against.append("Lack of central predominance")

        if features.bilateral_symmetry_score > 0.85:
            edema_score += 0.3
            edema_evidence_for.append(
                f"Bilateral symmetric pattern (symmetry: {features.bilateral_symmetry_score:.3f})"
            )

        if features.texture_entropy > 0.55:
            edema_score += 0.2
            edema_evidence_for.append("Heterogeneous texture - interstitial pattern with possible Kerley B lines")

        candidates["7. Edema"] = {
            "score": edema_score,
            "evidence_for": edema_evidence_for,
            "evidence_against": edema_evidence_against,
        }

        # Consolidation Lung: Dense, homogeneous
        consol_score = 0.0
        consol_evidence_for = []
        consol_evidence_against = []

        if features.texture_homogeneity > 0.50:
            consol_score += 0.5
            consol_evidence_for.append(
                f"Very dense homogeneous pattern (homogeneity: {features.texture_homogeneity:.3f})"
            )

        if features.opacity_score > 0.55:
            consol_score += 0.3
            consol_evidence_for.append(f"High opacity (score: {features.opacity_score:.3f})")

        candidates["3. Consolidation Lung"] = {
            "score": consol_score,
            "evidence_for": consol_evidence_for,
            "evidence_against": consol_evidence_against,
        }

        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)
        return self._format_differential(sorted_candidates)

    def _diff_cluster_b(self, features: CXRFeatures) -> Dict:
        """Differential diagnosis for Structural Changes cluster."""
        candidates = {}

        # Pneumothorax: Hyperlucency, visceral pleural line
        ptx_score = 0.0
        ptx_evidence_for = []
        ptx_evidence_against = []

        if features.opacity_score < 0.30:
            ptx_score += 0.5
            ptx_evidence_for.append(
                f"Hyperlucency (opacity: {features.opacity_score:.3f} <0.30) - too black appearance"
            )
            ptx_evidence_for.append("Absence of vascular markings in affected area")

        if features.bilateral_symmetry_score < 0.80:
            ptx_score += 0.3
            ptx_evidence_for.append(
                f"Unilateral (symmetry: {features.bilateral_symmetry_score:.3f}) - typical PTX presentation"
            )

        if features.texture_homogeneity > 0.40:
            ptx_score += 0.2
            ptx_evidence_for.append("Homogeneous dark region - pure air without lung tissue")

        candidates["6. Pneumothorax"] = {
            "score": ptx_score,
            "evidence_for": ptx_evidence_for,
            "evidence_against": ptx_evidence_against,
        }

        # Atelectasis: Volume loss, shift towards lesion
        atel_score = 0.0
        atel_evidence_for = []
        atel_evidence_against = []

        if features.bilateral_symmetry_score < 0.85:
            atel_score += 0.4
            atel_evidence_for.append(
                f"Asymmetric (symmetry: {features.bilateral_symmetry_score:.3f}) - unilateral volume loss"
            )

        if 0.35 < features.opacity_score < 0.50:
            atel_score += 0.4
            atel_evidence_for.append(
                f"Increased density with volume loss (opacity: {features.opacity_score:.3f})"
            )
            atel_evidence_for.append("Linear/wedge-shaped opacity pattern likely")

        candidates["4. Atelectasis"] = {
            "score": atel_score,
            "evidence_for": atel_evidence_for,
            "evidence_against": atel_evidence_against,
        }

        # Lung Cancer: Mass, nodule, spiculated margin
        cancer_score = 0.0
        cancer_evidence_for = []
        cancer_evidence_against = []

        if features.bilateral_symmetry_score < 0.80:
            cancer_score += 0.4
            cancer_evidence_for.append(
                f"Focal unilateral lesion (symmetry: {features.bilateral_symmetry_score:.3f})"
            )

        if 0.40 < features.opacity_score < 0.55:
            cancer_score += 0.3
            cancer_evidence_for.append("Moderate opacity increase - solitary nodule/mass likely")

        if features.texture_entropy > 0.50:
            cancer_score += 0.3
            cancer_evidence_for.append(
                f"Heterogeneous texture (entropy: {features.texture_entropy:.3f}) - possible spiculated margins"
            )

        candidates["2. Lungs Cancer"] = {
            "score": cancer_score,
            "evidence_for": cancer_evidence_for,
            "evidence_against": cancer_evidence_against,
        }

        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)
        return self._format_differential(sorted_candidates)

    def _format_differential(self, sorted_candidates: List[Tuple[str, Dict]]) -> Dict:
        """Format differential diagnosis results."""
        if not sorted_candidates:
            return {}

        primary = sorted_candidates[0]
        result = {
            "primary_candidate": primary[0],
            "primary_score": float(primary[1]["score"]),
            "evidence_for_primary": primary[1]["evidence_for"],
            "evidence_against_primary": primary[1]["evidence_against"],
        }

        if len(sorted_candidates) > 1:
            alternative = sorted_candidates[1]
            result["alternative_candidate"] = alternative[0]
            result["alternative_score"] = float(alternative[1]["score"])
            result["evidence_for_alternative"] = alternative[1]["evidence_for"]
            result["evidence_against_alternative"] = alternative[1]["evidence_against"]

        result["confidence_scores"] = {
            name: float(data["score"]) for name, data in sorted_candidates
        }

        return result

    def _generate_conclusion(
        self, level_1: Dict, level_2: Dict, level_3: Dict
    ) -> str:
        """Generate human-readable radiological conclusion."""
        category = level_1["category"]
        pattern = level_2.get("pattern_type", "Unknown pattern")
        primary = level_3.get("primary_candidate", "Unknown")
        primary_score = level_3.get("primary_score", 0.0)

        conclusion_parts = []

        conclusion_parts.append(
            f"CXR analysis indicates radiological findings consistent with {category}"
        )

        conclusion_parts.append(f"demonstrating {pattern.lower()}")

        if primary_score > 0.5:
            evidence_summary = level_3.get("evidence_for_primary", [])
            if evidence_summary:
                key_evidence = evidence_summary[0] if len(evidence_summary) > 0 else "multiple factors"
                conclusion_parts.append(
                    f"Primary radiological hypothesis is {primary} based on {key_evidence}"
                )
            else:
                conclusion_parts.append(f"Primary radiological hypothesis is {primary}")
        elif primary_score > 0.3:
            conclusion_parts.append(
                f"Features suggest possible {primary}, though confidence is moderate"
            )
        else:
            conclusion_parts.append(
                "However, radiological features are insufficient for specific disease differentiation"
            )

        alternative = level_3.get("alternative_candidate")
        alternative_score = level_3.get("alternative_score", 0.0)
        if alternative and abs(primary_score - alternative_score) < 0.2:
            conclusion_parts.append(
                f"Differential diagnosis should consider {alternative} as well"
            )

        conclusion_parts.append(
            "Clinical correlation with symptoms, labs, and physical exam strongly recommended."
        )

        return ". ".join(conclusion_parts) + "."

