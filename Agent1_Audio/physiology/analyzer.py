"""Hierarchical Physiological Analysis for Audio Features.

This module implements a 3-level clinical reasoning process:
- Level 1: Broad pathophysiological category (Infectious/Structural/Mass-Fluid)
- Level 2: Pattern recognition within category
- Level 3: Disease-specific differential diagnosis

Clinical intent: Provide interpretable, hypothesis-level evidence (not diagnosis).
"""

from dataclasses import asdict
from typing import Dict, List, Tuple

from Agent1_Audio.physiology.features import AudioFeatures


class HierarchicalPhysiologyAnalyzer:
    """Hierarchical audio-based physiological analysis engine."""

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
    CLUSTER_A = ["1. COVID-19", "5. Tuberculosis", "8. Pneumonia"]
    CLUSTER_B = ["4. Atelectasis", "6. Pneumothorax"]
    CLUSTER_C = ["2. Lungs Cancer", "3. Consolidation Lung", "7. Edema"]

    def analyze(self, features: AudioFeatures) -> Dict:
        """Perform hierarchical analysis on audio features.

        Args:
            features: Extracted AudioFeatures object

        Returns:
            Dictionary with hierarchical analysis results
        """
        # Level 1: Classify into broad category
        level_1_result = self._classify_level_1(features)

        # Level 2: Pattern recognition within category
        level_2_result = self._analyze_level_2(features, level_1_result["category"])

        # Level 3: Disease-specific differential
        level_3_result = self._differential_diagnosis(
            features, level_1_result["category"], level_2_result
        )

        # Generate physiological conclusion
        conclusion = self._generate_conclusion(
            level_1_result, level_2_result, level_3_result
        )

        return {
            "hierarchical_analysis": {
                "level_1": level_1_result,
                "level_2": level_2_result,
                "level_3": level_3_result,
                "physiological_conclusion": conclusion,
            },
            "raw_features": asdict(features),
        }

    def _classify_level_1(self, features: AudioFeatures) -> Dict:
        """Level 1: Classify into broad pathophysiological category."""
        scores = {
            "Cluster A: Infectious/Inflammatory": 0.0,
            "Cluster B: Structural/Pleural": 0.0,
            "Cluster C: Mass/Fluid": 0.0,
        }
        evidence = []

        # Cluster A indicators: high cough rate + burstiness
        if features.cough_rate_per_min > 5:
            scores["Cluster A: Infectious/Inflammatory"] += 0.4
            evidence.append(
                f"High cough rate: {features.cough_rate_per_min:.1f}/min (suggests active inflammatory process)"
            )

        if features.temporal_burstiness > 0.5:
            scores["Cluster A: Infectious/Inflammatory"] += 0.3
            evidence.append(
                f"High temporal burstiness: {features.temporal_burstiness:.2f} (episodic pattern typical of infection)"
            )

        # Cluster B indicators: very low energy + minimal high-freq
        if features.high_freq_energy_ratio < 0.1:
            scores["Cluster B: Structural/Pleural"] += 0.5
            evidence.append(
                f"Severely reduced high-frequency content: {features.high_freq_energy_ratio:.3f} (suggests diminished ventilation)"
            )

        if features.cough_rate_per_min < 2:
            scores["Cluster B: Structural/Pleural"] += 0.3
            evidence.append(
                f"Minimal cough activity: {features.cough_rate_per_min:.1f}/min (consistent with structural issue)"
            )

        # Cluster C indicators: moderate features, spectral changes
        if 0.1 <= features.high_freq_energy_ratio <= 0.25:
            scores["Cluster C: Mass/Fluid"] += 0.3

        if 2 <= features.cough_rate_per_min <= 5:
            scores["Cluster C: Mass/Fluid"] += 0.2

        if features.spectral_centroid_hz < 1500 or features.spectral_bandwidth_hz > 1000:
            scores["Cluster C: Mass/Fluid"] += 0.3
            evidence.append(
                f"Altered spectral characteristics (centroid: {features.spectral_centroid_hz:.0f} Hz, bandwidth: {features.spectral_bandwidth_hz:.0f} Hz)"
            )

        # Normalize and select winner
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

    def _analyze_level_2(self, features: AudioFeatures, category: str) -> Dict:
        """Level 2: Pattern recognition within the identified category."""
        if "Cluster A" in category:
            return self._level_2_cluster_a(features)
        elif "Cluster B" in category:
            return self._level_2_cluster_b(features)
        elif "Cluster C" in category:
            return self._level_2_cluster_c(features)
        else:
            return {"pattern_type": "Unknown", "features": {}}

    def _level_2_cluster_a(self, features: AudioFeatures) -> Dict:
        """Level 2 analysis for Infectious/Inflammatory cluster."""
        pattern_type = "Undetermined inflammatory pattern"
        pattern_features = {}

        # Check for wet vs dry cough signature
        if features.spectral_centroid_hz > 2000 and features.high_freq_energy_ratio > 0.3:
            pattern_type = "Dry cough signature (viral-like)"
            pattern_features["cough_productivity"] = "Dry (high-frequency dominant)"
        elif features.spectral_centroid_hz < 1500:
            pattern_type = "Wet/productive cough signature (bacterial-like)"
            pattern_features["cough_productivity"] = "Wet (low-frequency resonance)"
        else:
            pattern_type = "Mixed cough pattern"
            pattern_features["cough_productivity"] = "Intermediate"

        pattern_features["spectral_centroid"] = f"{features.spectral_centroid_hz:.0f} Hz"
        pattern_features["temporal_pattern"] = (
            "Bursty" if features.temporal_burstiness > 0.6 else "Regular"
        )

        return {"pattern_type": pattern_type, "features": pattern_features}

    def _level_2_cluster_b(self, features: AudioFeatures) -> Dict:
        """Level 2 analysis for Structural/Pleural cluster."""
        pattern_features = {
            "high_freq_energy_ratio": f"{features.high_freq_energy_ratio:.3f}",
            "cough_rate": f"{features.cough_rate_per_min:.1f}/min",
        }

        if features.high_freq_energy_ratio < 0.05:
            pattern_type = "Severely diminished ventilation (pneumothorax-like)"
        else:
            pattern_type = "Reduced ventilation with some preserved sound"

        return {"pattern_type": pattern_type, "features": pattern_features}

    def _level_2_cluster_c(self, features: AudioFeatures) -> Dict:
        """Level 2 analysis for Mass/Fluid cluster."""
        pattern_features = {
            "spectral_centroid": f"{features.spectral_centroid_hz:.0f} Hz",
            "spectral_bandwidth": f"{features.spectral_bandwidth_hz:.0f} Hz",
        }

        if features.high_freq_energy_ratio > 0.3:
            pattern_type = "Fine crackles pattern (fluid-like)"
            pattern_features["suspected_content"] = "Fluid accumulation"
        elif features.spectral_centroid_hz < 1300:
            pattern_type = "Dense/consolidated pattern"
            pattern_features["suspected_content"] = "Solid/consolidated tissue"
        else:
            pattern_type = "Mass effect with variable density"
            pattern_features["suspected_content"] = "Mass or focal lesion"

        return {"pattern_type": pattern_type, "features": pattern_features}

    def _differential_diagnosis(
        self, features: AudioFeatures, category: str, level_2: Dict
    ) -> Dict:
        """Level 3: Disease-specific differential diagnosis."""
        if "Cluster A" in category:
            return self._diff_cluster_a(features)
        elif "Cluster B" in category:
            return self._diff_cluster_b(features)
        elif "Cluster C" in category:
            return self._diff_cluster_c(features)
        else:
            return {}

    def _diff_cluster_a(self, features: AudioFeatures) -> Dict:
        """Differential diagnosis for Infectious cluster.
        
        Based on design doc:
        - COVID-19: Dry cough, short explosive, high harmonics (>1kHz), energy concentrated high
        - Pneumonia: Wet/productive, coarse crackles, strong low-freq energy (200-500Hz)
        - TB: Chronic pattern, repetitive bouts, possible wheeze
        """
        candidates = {}

        # COVID-19: High-Harmonic "Dry" Signature
        # - Short explosive phase duration
        # - Energy concentrated in higher frequencies (>1kHz)
        covid_score = 0.0
        covid_evidence_for = []
        covid_evidence_against = []

        # Key: High spectral centroid (>1000Hz, ideally >2000Hz for dry cough)
        if features.spectral_centroid_hz > 2000:
            covid_score += 0.5
            covid_evidence_for.append(
                f"High-harmonic dry signature: spectral centroid {features.spectral_centroid_hz:.0f} Hz (>1kHz threshold)"
            )
        elif features.spectral_centroid_hz > 1000:
            covid_score += 0.2
            covid_evidence_for.append(
                f"Moderate high-frequency content: {features.spectral_centroid_hz:.0f} Hz"
            )
        else:
            covid_evidence_against.append(
                f"Low spectral centroid ({features.spectral_centroid_hz:.0f} Hz) - inconsistent with dry viral cough"
            )

        # High-frequency energy dominance
        if features.high_freq_energy_ratio > 0.3:
            covid_score += 0.3
            covid_evidence_for.append(
                f"Energy concentrated in higher frequencies (ratio: {features.high_freq_energy_ratio:.3f})"
            )
        else:
            covid_evidence_against.append(
                f"Low high-frequency energy ratio ({features.high_freq_energy_ratio:.3f}) - suggests secretions present"
            )

        # Short explosive pattern (low burstiness = more regular)
        if features.temporal_burstiness < 0.5:
            covid_score += 0.2
            covid_evidence_for.append(
                f"Short explosive phase pattern (burstiness: {features.temporal_burstiness:.2f})"
            )

        candidates["1. COVID-19"] = {
            "score": covid_score,
            "evidence_for": covid_evidence_for,
            "evidence_against": covid_evidence_against,
        }

        # Bacterial Pneumonia: Coarse Crackles + Wet/Productive Signature
        # - Strong Low-freq Energy (200-500Hz) from mucus
        # - Coarse Crackles (거친 수포음)
        pneumonia_score = 0.0
        pneumonia_evidence_for = []
        pneumonia_evidence_against = []

        # Key: Low spectral centroid indicating mucus resonance
        if features.spectral_centroid_hz < 1000:
            pneumonia_score += 0.5
            pneumonia_evidence_for.append(
                f"Strong low-frequency mucus resonance: centroid {features.spectral_centroid_hz:.0f} Hz (<1kHz)"
            )
            if features.spectral_centroid_hz < 500:
                pneumonia_score += 0.1
                pneumonia_evidence_for.append("Very strong low-band energy (200-500Hz range)")
        elif features.spectral_centroid_hz < 1500:
            pneumonia_score += 0.2
            pneumonia_evidence_for.append(f"Moderate low-frequency content: {features.spectral_centroid_hz:.0f} Hz")
        else:
            pneumonia_evidence_against.append(
                f"High spectral centroid ({features.spectral_centroid_hz:.0f} Hz) - less typical for bacterial"
            )

        # Low high-frequency energy ratio (wet signature)
        if features.high_freq_energy_ratio < 0.2:
            pneumonia_score += 0.3
            pneumonia_evidence_for.append(
                f"Wet/productive signature: low high-freq ratio ({features.high_freq_energy_ratio:.3f})"
            )

        # Coarse crackles pattern (moderate to high cough rate)
        if features.cough_rate_per_min > 5:
            pneumonia_score += 0.2
            pneumonia_evidence_for.append(
                f"Coarse crackles pattern: cough rate {features.cough_rate_per_min:.1f}/min"
            )

        candidates["8. Pneumonia"] = {
            "score": pneumonia_score,
            "evidence_for": pneumonia_evidence_for,
            "evidence_against": pneumonia_evidence_against,
        }

        # Tuberculosis: Chronic Repetitive Pattern
        # - Repetitive bouts (high burstiness)
        # - Possible stridor/wheeze during inspiration
        # - Chronic irritation
        tb_score = 0.0
        tb_evidence_for = []
        tb_evidence_against = []

        # Key: Very high temporal burstiness (episodic bouts)
        if features.temporal_burstiness > 0.7:
            tb_score += 0.5
            tb_evidence_for.append(
                f"Chronic repetitive bout pattern: burstiness {features.temporal_burstiness:.2f} (>0.7 threshold)"
            )
        elif features.temporal_burstiness > 0.5:
            tb_score += 0.2
            tb_evidence_for.append(f"Episodic pattern detected: burstiness {features.temporal_burstiness:.2f}")
        else:
            tb_evidence_against.append(
                f"Low burstiness ({features.temporal_burstiness:.2f}) - less typical for chronic TB"
            )

        # High cough rate (chronic irritation)
        if features.cough_rate_per_min > 8:
            tb_score += 0.3
            tb_evidence_for.append(
                f"Very high cough rate ({features.cough_rate_per_min:.1f}/min) - chronic airway irritation"
            )
        elif features.cough_rate_per_min > 5:
            tb_score += 0.1
            tb_evidence_for.append(f"Elevated cough rate: {features.cough_rate_per_min:.1f}/min")

        # Broad spectral bandwidth (mixed pathology, possible wheeze)
        if features.spectral_bandwidth_hz > 1200:
            tb_score += 0.2
            tb_evidence_for.append(
                f"Broad spectral bandwidth ({features.spectral_bandwidth_hz:.0f} Hz) - possible stridor/wheeze component"
            )

        candidates["5. Tuberculosis"] = {
            "score": tb_score,
            "evidence_for": tb_evidence_for,
            "evidence_against": tb_evidence_against,
        }

        # Determine primary and alternative
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)

        return self._format_differential(sorted_candidates)

    def _diff_cluster_b(self, features: AudioFeatures) -> Dict:
        """Differential diagnosis for Structural cluster.
        
        Based on design doc:
        - Pneumothorax: "Silent Chest" - complete absence, high frequency cutoff, no crackles
        - Atelectasis: Diminished + late inspiratory crackles (re-expansion sounds)
        """
        candidates = {}

        # Pneumothorax: "Silent Chest" / High Frequency Cutoff
        # - Significant RMS Energy Drop
        # - Complete absence of breath sounds
        # - No crackles (pure silence)
        ptx_score = 0.0
        ptx_evidence_for = []
        ptx_evidence_against = []

        # Key: Severely reduced high-frequency content (>2kHz cutoff)
        if features.high_freq_energy_ratio < 0.05:
            ptx_score += 0.6
            ptx_evidence_for.append(
                f"'Silent chest' signature: severely reduced high-frequency content ({features.high_freq_energy_ratio:.3f} <0.05)"
            )
            ptx_evidence_for.append("Complete absence of breath sounds in affected area")
        elif features.high_freq_energy_ratio < 0.1:
            ptx_score += 0.3
            ptx_evidence_for.append(f"Significant high-frequency cutoff: {features.high_freq_energy_ratio:.3f}")
        else:
            ptx_evidence_against.append(
                f"Preserved high-frequency content ({features.high_freq_energy_ratio:.3f}) - inconsistent with pneumothorax"
            )

        # Minimal to no cough (pure silence, not productive)
        if features.cough_rate_per_min < 1:
            ptx_score += 0.3
            ptx_evidence_for.append(
                f"Minimal cough activity ({features.cough_rate_per_min:.1f}/min) - pure silence pattern"
            )
        elif features.cough_rate_per_min < 2:
            ptx_score += 0.1

        # No crackles (low spectral variance expected)
        if features.spectral_bandwidth_hz < 600:
            ptx_evidence_for.append("Narrow bandwidth - no adventitious sounds (no crackles)")

        candidates["6. Pneumothorax"] = {
            "score": ptx_score,
            "evidence_for": ptx_evidence_for,
            "evidence_against": ptx_evidence_against,
        }

        # Atelectasis: Diminished + Re-expansion Sounds
        # - Reduced amplitude (less severe than Pneumothorax)
        # - Late Inspiratory Crackles (collapsed alveoli popping open)
        atel_score = 0.0
        atel_evidence_for = []
        atel_evidence_against = []

        # Key: Reduced but PRESENT sound (not complete silence)
        if 0.05 <= features.high_freq_energy_ratio <= 0.15:
            atel_score += 0.5
            atel_evidence_for.append(
                f"Diminished but present ventilation: high-freq ratio {features.high_freq_energy_ratio:.3f} (0.05-0.15 range)"
            )
        elif 0.1 <= features.high_freq_energy_ratio <= 0.20:
            atel_score += 0.3
            atel_evidence_for.append(f"Reduced ventilation sounds: {features.high_freq_energy_ratio:.3f}")
        else:
            atel_evidence_against.append("High-frequency content outside typical atelectasis range")

        # Some cough activity (re-expansion attempts)
        if 1 <= features.cough_rate_per_min <= 4:
            atel_score += 0.3
            atel_evidence_for.append(
                f"Moderate cough activity ({features.cough_rate_per_min:.1f}/min) - possible re-expansion efforts"
            )

        # Late inspiratory crackles (indicated by some spectral variance)
        if features.spectral_bandwidth_hz > 600:
            atel_score += 0.2
            atel_evidence_for.append(
                f"Spectral variance present ({features.spectral_bandwidth_hz:.0f} Hz) - possible fine re-expansion crackles"
            )

        candidates["4. Atelectasis"] = {
            "score": atel_score,
            "evidence_for": atel_evidence_for,
            "evidence_against": atel_evidence_against,
        }

        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)
        return self._format_differential(sorted_candidates)

    def _diff_cluster_c(self, features: AudioFeatures) -> Dict:
        """Differential diagnosis for Mass/Fluid cluster.
        
        Based on design doc:
        - Edema: "Velcro" fine crackles - high-pitched, end inspiration, uniform pattern
        - Lung Cancer: Localized monophonic wheeze - single pitch, constant across cycles
        - Consolidation: Dense, low centroid
        """
        candidates = {}

        # Edema: "Velcro" Fine Crackles
        # - High-pitched, fine, continuous crackles
        # - Predominant at the end of inspiration
        # - Uniform pattern (unlike irregular pneumonia crackles)
        edema_score = 0.0
        edema_evidence_for = []
        edema_evidence_against = []

        # Key: High-frequency content (fine crackles)
        if features.high_freq_energy_ratio > 0.3:
            edema_score += 0.5
            edema_evidence_for.append(
                f"'Velcro' fine crackles signature: high-frequency content ({features.high_freq_energy_ratio:.3f} >0.3)"
            )
            edema_evidence_for.append("High-pitched fine crackles at end inspiration")
        else:
            edema_evidence_against.append(
                f"Low high-frequency content ({features.high_freq_energy_ratio:.3f}) - inconsistent with fine crackles"
            )

        # Uniform pattern (moderate regularity)
        if 0.3 < features.temporal_burstiness < 0.6:
            edema_score += 0.3
            edema_evidence_for.append(
                f"Uniform crackle pattern: temporal burstiness {features.temporal_burstiness:.2f} (moderate regularity)"
            )
        elif features.temporal_burstiness <= 0.3:
            edema_score += 0.2
            edema_evidence_for.append("Very uniform pattern - consistent with fluid accumulation")

        # Moderate cough rate
        if 3 <= features.cough_rate_per_min <= 7:
            edema_score += 0.2
            edema_evidence_for.append(f"Moderate cough rate ({features.cough_rate_per_min:.1f}/min)")

        candidates["7. Edema"] = {
            "score": edema_score,
            "evidence_for": edema_evidence_for,
            "evidence_against": edema_evidence_against,
        }

        # Consolidation: Dense, Low Centroid
        # - Very low spectral centroid (dense tissue)
        # - Low high-frequency ratio
        consol_score = 0.0
        consol_evidence_for = []
        consol_evidence_against = []

        # Key: Very low spectral centroid (dense consolidated pattern)
        if features.spectral_centroid_hz < 1000:
            consol_score += 0.5
            consol_evidence_for.append(
                f"Dense consolidated pattern: very low centroid {features.spectral_centroid_hz:.0f} Hz (<1kHz)"
            )
        elif features.spectral_centroid_hz < 1300:
            consol_score += 0.3
            consol_evidence_for.append(f"Low spectral centroid: {features.spectral_centroid_hz:.0f} Hz")
        else:
            consol_evidence_against.append("High spectral centroid - less typical for dense consolidation")

        # Low high-frequency ratio (dense tissue absorbs high freq)
        if features.high_freq_energy_ratio < 0.2:
            consol_score += 0.3
            consol_evidence_for.append(
                f"Low high-frequency ratio ({features.high_freq_energy_ratio:.3f}) - dense tissue effect"
            )

        # Moderate to high cough rate
        if features.cough_rate_per_min > 4:
            consol_score += 0.2
            consol_evidence_for.append(f"Elevated cough rate: {features.cough_rate_per_min:.1f}/min")

        candidates["3. Consolidation Lung"] = {
            "score": consol_score,
            "evidence_for": consol_evidence_for,
            "evidence_against": consol_evidence_against,
        }

        # Lung Cancer: Localized Monophonic Wheeze
        # - Single-pitch whistling sound (fixed obstruction)
        # - Constant across breath cycles
        # - Narrow spectral bandwidth
        cancer_score = 0.0
        cancer_evidence_for = []
        cancer_evidence_against = []

        # Key: Narrow spectral bandwidth (single-pitch wheeze)
        if features.spectral_bandwidth_hz < 600:
            cancer_score += 0.4
            cancer_evidence_for.append(
                f"Localized monophonic wheeze: narrow bandwidth {features.spectral_bandwidth_hz:.0f} Hz (<600Hz)"
            )
            cancer_evidence_for.append("Single-pitch whistling pattern - fixed obstruction")
        elif features.spectral_bandwidth_hz < 800:
            cancer_score += 0.2
            cancer_evidence_for.append(f"Relatively narrow bandwidth: {features.spectral_bandwidth_hz:.0f} Hz")

        # Mid-range spectral centroid (not too high, not too low)
        if 1500 < features.spectral_centroid_hz < 2500:
            cancer_score += 0.3
            cancer_evidence_for.append(
                f"Mid-range spectral centroid ({features.spectral_centroid_hz:.0f} Hz) - typical wheeze range"
            )

        # Lower cough rate (mass effect, not productive inflammation)
        if features.cough_rate_per_min < 4:
            cancer_score += 0.3
            cancer_evidence_for.append(
                f"Lower cough rate ({features.cough_rate_per_min:.1f}/min) - mass effect pattern"
            )

        # Constant pattern (low burstiness)
        if features.temporal_burstiness < 0.4:
            cancer_evidence_for.append(
                f"Constant pattern across cycles: burstiness {features.temporal_burstiness:.2f}"
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

        # Confidence scores for all candidates
        result["confidence_scores"] = {
            name: float(data["score"]) for name, data in sorted_candidates
        }

        return result

    def _generate_conclusion(
        self, level_1: Dict, level_2: Dict, level_3: Dict
    ) -> str:
        """Generate human-readable physiological conclusion."""
        category = level_1["category"]
        pattern = level_2.get("pattern_type", "Unknown pattern")
        primary = level_3.get("primary_candidate", "Unknown")
        primary_score = level_3.get("primary_score", 0.0)

        conclusion_parts = []

        # Start with category
        conclusion_parts.append(
            f"Audio analysis indicates pathophysiology consistent with {category}"
        )

        # Add pattern description
        conclusion_parts.append(f"with characteristic {pattern.lower()}")

        # Add primary candidate if score is reasonable
        if primary_score > 0.5:
            evidence_summary = level_3.get("evidence_for_primary", [])
            if evidence_summary:
                key_evidence = evidence_summary[0] if len(evidence_summary) > 0 else "multiple factors"
                conclusion_parts.append(
                    f"Primary hypothesis is {primary} based on {key_evidence}"
                )
            else:
                conclusion_parts.append(f"Primary hypothesis is {primary}")
        elif primary_score > 0.3:
            conclusion_parts.append(
                f"Features suggest possible {primary}, though confidence is moderate"
            )
        else:
            conclusion_parts.append(
                "However, audio features are insufficient for specific disease differentiation"
            )

        # Add alternative if present and scores are close
        alternative = level_3.get("alternative_candidate")
        alternative_score = level_3.get("alternative_score", 0.0)
        if alternative and abs(primary_score - alternative_score) < 0.2:
            conclusion_parts.append(
                f"Differential diagnosis should consider {alternative} as well"
            )

        # Final caveat
        conclusion_parts.append(
            "Clinical correlation and imaging confirmation strongly recommended."
        )

        return ". ".join(conclusion_parts) + "."

