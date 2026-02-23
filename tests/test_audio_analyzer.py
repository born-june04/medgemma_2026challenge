#!/usr/bin/env python3
"""Test script for hierarchical audio physiology analyzer."""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Agent1_Audio.physiology.features import AudioFeatures, extract_audio_features
from Agent1_Audio.physiology.analyzer import HierarchicalPhysiologyAnalyzer


def test_analyzer_with_sample_features():
    """Test analyzer with various sample feature patterns."""
    
    analyzer = HierarchicalPhysiologyAnalyzer()
    
    # Test Case 1: COVID-19-like pattern (dry cough)
    print("=" * 80)
    print("TEST CASE 1: COVID-19-like Pattern (Dry Cough)")
    print("=" * 80)
    covid_features = AudioFeatures(
        cough_rate_per_min=7.5,
        inter_cough_interval_mean=8.0,
        inter_cough_interval_std=3.5,
        spectral_centroid_hz=2300.0,  # High (dry)
        spectral_bandwidth_hz=1000.0,
        high_freq_energy_ratio=0.35,  # High
        temporal_burstiness=0.42,  # Regular
    )
    
    result = analyzer.analyze(covid_features)
    print(json.dumps(result, indent=2))
    print("\n")
    
    # Test Case 2: Pneumonia-like pattern (wet cough)
    print("=" * 80)
    print("TEST CASE 2: Pneumonia-like Pattern (Wet/Productive Cough)")
    print("=" * 80)
    pneumonia_features = AudioFeatures(
        cough_rate_per_min=9.0,
        inter_cough_interval_mean=6.5,
        inter_cough_interval_std=2.8,
        spectral_centroid_hz=850.0,  # Low (mucus)
        spectral_bandwidth_hz=700.0,
        high_freq_energy_ratio=0.15,  # Low
        temporal_burstiness=0.45,
    )
    
    result = analyzer.analyze(pneumonia_features)
    print(json.dumps(result, indent=2))
    print("\n")
    
    # Test Case 3: TB-like pattern (chronic, bursty)
    print("=" * 80)
    print("TEST CASE 3: Tuberculosis-like Pattern (Chronic, Episodic)")
    print("=" * 80)
    tb_features = AudioFeatures(
        cough_rate_per_min=11.5,
        inter_cough_interval_mean=5.2,
        inter_cough_interval_std=4.5,  # High variance
        spectral_centroid_hz=1500.0,
        spectral_bandwidth_hz=1300.0,  # Broad
        high_freq_energy_ratio=0.22,
        temporal_burstiness=0.87,  # Very high
    )
    
    result = analyzer.analyze(tb_features)
    print(json.dumps(result, indent=2))
    print("\n")
    
    # Test Case 4: Pneumothorax-like pattern (silent chest)
    print("=" * 80)
    print("TEST CASE 4: Pneumothorax-like Pattern (Silent Chest)")
    print("=" * 80)
    ptx_features = AudioFeatures(
        cough_rate_per_min=0.5,
        inter_cough_interval_mean=0.0,
        inter_cough_interval_std=0.0,
        spectral_centroid_hz=800.0,
        spectral_bandwidth_hz=400.0,
        high_freq_energy_ratio=0.03,  # Very low
        temporal_burstiness=0.1,
    )
    
    result = analyzer.analyze(ptx_features)
    print(json.dumps(result, indent=2))
    print("\n")
    
    # Test Case 5: Edema-like pattern (fine crackles)
    print("=" * 80)
    print("TEST CASE 5: Edema-like Pattern (Fine Crackles)")
    print("=" * 80)
    edema_features = AudioFeatures(
        cough_rate_per_min=5.0,
        inter_cough_interval_mean=12.0,
        inter_cough_interval_std=4.0,
        spectral_centroid_hz=1800.0,
        spectral_bandwidth_hz=900.0,
        high_freq_energy_ratio=0.34,  # High (fine crackles)
        temporal_burstiness=0.35,  # Uniform
    )
    
    result = analyzer.analyze(edema_features)
    print(json.dumps(result, indent=2))
    print("\n")
    
    # Test Case 6: Lung Cancer-like pattern (monophonic wheeze)
    print("=" * 80)
    print("TEST CASE 6: Lung Cancer-like Pattern (Localized Wheeze)")
    print("=" * 80)
    cancer_features = AudioFeatures(
        cough_rate_per_min=2.5,
        inter_cough_interval_mean=24.0,
        inter_cough_interval_std=8.0,
        spectral_centroid_hz=1900.0,  # Mid-range
        spectral_bandwidth_hz=450.0,  # Narrow (monophonic)
        high_freq_energy_ratio=0.18,
        temporal_burstiness=0.25,  # Constant
    )
    
    result = analyzer.analyze(cancer_features)
    print(json.dumps(result, indent=2))
    print("\n")


def test_with_real_audio(audio_path: str):
    """Test analyzer with real audio file."""
    print("=" * 80)
    print(f"REAL AUDIO TEST: {audio_path}")
    print("=" * 80)
    
    try:
        # Extract features
        features = extract_audio_features(audio_path)
        print("\nExtracted Features:")
        print(f"  Cough rate: {features.cough_rate_per_min:.1f}/min")
        print(f"  Inter-cough interval: {features.inter_cough_interval_mean:.2f}s ± {features.inter_cough_interval_std:.2f}s")
        print(f"  Spectral centroid: {features.spectral_centroid_hz:.0f} Hz")
        print(f"  Spectral bandwidth: {features.spectral_bandwidth_hz:.0f} Hz")
        print(f"  High-freq energy ratio: {features.high_freq_energy_ratio:.3f}")
        print(f"  Temporal burstiness: {features.temporal_burstiness:.2f}")
        print()
        
        # Analyze
        analyzer = HierarchicalPhysiologyAnalyzer()
        result = analyzer.analyze(features)
        
        print("\nHierarchical Analysis:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with real audio file
        audio_path = sys.argv[1]
        test_with_real_audio(audio_path)
    else:
        # Test with synthetic patterns
        test_analyzer_with_sample_features()

