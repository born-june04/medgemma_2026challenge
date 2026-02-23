#!/usr/bin/env python3
"""Test script for hierarchical CXR physiology analyzer."""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Agent2_Image.physiology.features import CXRFeatures, extract_cxr_features
from Agent2_Image.physiology.analyzer import HierarchicalCXRAnalyzer


def test_analyzer_with_sample_features():
    """Test analyzer with various sample CXR feature patterns."""
    
    analyzer = HierarchicalCXRAnalyzer()
    
    # Test Case 1: COVID-19-like pattern (peripheral GGO)
    print("=" * 80)
    print("TEST CASE 1: COVID-19-like Pattern (Peripheral GGO)")
    print("=" * 80)
    covid_features = CXRFeatures(
        mean_intensity=0.48,
        opacity_score=0.55,
        upper_left_intensity=0.50,
        upper_right_intensity=0.52,
        mid_left_intensity=0.48,
        mid_right_intensity=0.50,
        lower_left_intensity=0.45,
        lower_right_intensity=0.47,
        apical_predominance_score=1.08,  # Slightly higher upper
        peripheral_predominance_score=1.45,  # Peripheral > central
        bilateral_symmetry_score=0.92,  # Bilateral
        texture_entropy=0.68,  # High (GGO)
        texture_homogeneity=0.35,
        central_intensity=0.42,
        peripheral_intensity=0.61,
    )
    
    result = analyzer.analyze(covid_features)
    print(json.dumps(result, indent=2))
    print("\n")
    
    # Test Case 2: Pneumonia-like pattern (lobar consolidation)
    print("=" * 80)
    print("TEST CASE 2: Pneumonia-like Pattern (Lobar Consolidation)")
    print("=" * 80)
    pneumonia_features = CXRFeatures(
        mean_intensity=0.52,
        opacity_score=0.58,
        upper_left_intensity=0.45,
        upper_right_intensity=0.68,  # Asymmetric - right upper lobe
        mid_left_intensity=0.42,
        mid_right_intensity=0.65,
        lower_left_intensity=0.40,
        lower_right_intensity=0.48,
        apical_predominance_score=1.25,
        peripheral_predominance_score=0.95,  # Focal, not peripheral
        bilateral_symmetry_score=0.72,  # Asymmetric
        texture_entropy=0.45,
        texture_homogeneity=0.52,  # High (dense consolidation)
        central_intensity=0.50,
        peripheral_intensity=0.48,
    )
    
    result = analyzer.analyze(pneumonia_features)
    print(json.dumps(result, indent=2))
    print("\n")
    
    # Test Case 3: TB-like pattern (apical predominance)
    print("=" * 80)
    print("TEST CASE 3: Tuberculosis-like Pattern (Apical Predominance)")
    print("=" * 80)
    tb_features = CXRFeatures(
        mean_intensity=0.50,
        opacity_score=0.54,
        upper_left_intensity=0.72,  # Very high apex
        upper_right_intensity=0.68,
        mid_left_intensity=0.45,
        mid_right_intensity=0.42,
        lower_left_intensity=0.38,
        lower_right_intensity=0.35,
        apical_predominance_score=1.75,  # Strong apical
        peripheral_predominance_score=1.10,
        bilateral_symmetry_score=0.78,  # Somewhat asymmetric
        texture_entropy=0.62,  # Heterogeneous (cavitation)
        texture_homogeneity=0.38,
        central_intensity=0.45,
        peripheral_intensity=0.52,
    )
    
    result = analyzer.analyze(tb_features)
    print(json.dumps(result, indent=2))
    print("\n")
    
    # Test Case 4: Pneumothorax-like pattern (hyperlucency)
    print("=" * 80)
    print("TEST CASE 4: Pneumothorax-like Pattern (Hyperlucency)")
    print("=" * 80)
    ptx_features = CXRFeatures(
        mean_intensity=0.28,
        opacity_score=0.25,  # Very low (black)
        upper_left_intensity=0.35,
        upper_right_intensity=0.18,  # Very dark right
        mid_left_intensity=0.32,
        mid_right_intensity=0.15,
        lower_left_intensity=0.30,
        lower_right_intensity=0.20,
        apical_predominance_score=1.10,
        peripheral_predominance_score=0.85,
        bilateral_symmetry_score=0.62,  # Very asymmetric
        texture_entropy=0.35,
        texture_homogeneity=0.48,  # Homogeneous dark
        central_intensity=0.28,
        peripheral_intensity=0.22,
    )
    
    result = analyzer.analyze(ptx_features)
    print(json.dumps(result, indent=2))
    print("\n")
    
    # Test Case 5: Edema-like pattern (central/bat-wing)
    print("=" * 80)
    print("TEST CASE 5: Edema-like Pattern (Central/Bat-wing)")
    print("=" * 80)
    edema_features = CXRFeatures(
        mean_intensity=0.51,
        opacity_score=0.56,
        upper_left_intensity=0.48,
        upper_right_intensity=0.50,
        mid_left_intensity=0.52,
        mid_right_intensity=0.54,
        lower_left_intensity=0.55,
        lower_right_intensity=0.53,
        apical_predominance_score=0.92,  # Lower predominance
        peripheral_predominance_score=0.62,  # Central predominance
        bilateral_symmetry_score=0.94,  # Very symmetric
        texture_entropy=0.58,  # Heterogeneous (Kerley B)
        texture_homogeneity=0.40,
        central_intensity=0.65,  # High central
        peripheral_intensity=0.40,
    )
    
    result = analyzer.analyze(edema_features)
    print(json.dumps(result, indent=2))
    print("\n")
    
    # Test Case 6: Lung Cancer-like pattern (focal mass)
    print("=" * 80)
    print("TEST CASE 6: Lung Cancer-like Pattern (Focal Mass)")
    print("=" * 80)
    cancer_features = CXRFeatures(
        mean_intensity=0.44,
        opacity_score=0.48,
        upper_left_intensity=0.42,
        upper_right_intensity=0.65,  # Focal lesion right upper
        mid_left_intensity=0.38,
        mid_right_intensity=0.42,
        lower_left_intensity=0.35,
        lower_right_intensity=0.38,
        apical_predominance_score=1.30,
        peripheral_predominance_score=0.88,
        bilateral_symmetry_score=0.68,  # Unilateral
        texture_entropy=0.58,  # Heterogeneous (spiculated)
        texture_homogeneity=0.42,
        central_intensity=0.45,
        peripheral_intensity=0.42,
    )
    
    result = analyzer.analyze(cancer_features)
    print(json.dumps(result, indent=2))
    print("\n")


def test_with_real_cxr(image_path: str):
    """Test analyzer with real CXR image."""
    print("=" * 80)
    print(f"REAL CXR TEST: {image_path}")
    print("=" * 80)
    
    try:
        # Extract features
        features = extract_cxr_features(image_path)
        print("\nExtracted Features:")
        print(f"  Mean intensity: {features.mean_intensity:.3f}")
        print(f"  Opacity score: {features.opacity_score:.3f}")
        print(f"  Apical predominance: {features.apical_predominance_score:.2f}")
        print(f"  Peripheral predominance: {features.peripheral_predominance_score:.2f}")
        print(f"  Bilateral symmetry: {features.bilateral_symmetry_score:.2f}")
        print(f"  Texture entropy: {features.texture_entropy:.3f}")
        print(f"  Texture homogeneity: {features.texture_homogeneity:.3f}")
        print()
        
        # Analyze
        analyzer = HierarchicalCXRAnalyzer()
        result = analyzer.analyze(features)
        
        print("\nHierarchical CXR Analysis:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error processing CXR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with real CXR image
        image_path = sys.argv[1]
        test_with_real_cxr(image_path)
    else:
        # Test with synthetic patterns
        test_analyzer_with_sample_features()

