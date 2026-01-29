"""CXR (Chest X-Ray) feature extraction for hierarchical physiological analysis.

This module extracts interpretable radiological features from CXR images:
- Zonal intensity analysis (Upper/Mid/Lower x Left/Right)
- Central vs Peripheral distribution
- Texture characteristics (entropy, homogeneity)
- Global metrics (mean intensity, opacity)

Clinical intent: quantify visual patterns to reason about pathology.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from PIL import Image


@dataclass
class CXRFeatures:
    """Interpretable CXR features for physiological hypotheses."""
    
    # Global metrics
    mean_intensity: float  # Overall brightness (0-1 normalized)
    opacity_score: float  # How "white" the lungs are
    
    # Zonal intensity (0-1 normalized)
    upper_left_intensity: float
    upper_right_intensity: float
    mid_left_intensity: float
    mid_right_intensity: float
    lower_left_intensity: float
    lower_right_intensity: float
    
    # Derived zonal metrics
    apical_predominance_score: float  # Upper vs Lower ratio
    peripheral_predominance_score: float  # Outer vs Central ratio
    bilateral_symmetry_score: float  # Left vs Right similarity
    
    # Texture metrics
    texture_entropy: float  # Randomness/complexity
    texture_homogeneity: float  # Uniformity
    
    # Distribution pattern
    central_intensity: float  # Central region
    peripheral_intensity: float  # Peripheral region


def extract_cxr_features(image_path: str) -> CXRFeatures:
    """Extract interpretable CXR features for hierarchical analysis.
    
    Args:
        image_path: Path to CXR image file
        
    Returns:
        CXRFeatures object with extracted metrics
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('L')  # Grayscale
    img_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Extract basic metrics
    mean_intensity = float(np.mean(img_array))
    opacity_score = _calculate_opacity_score(img_array)
    
    # Zonal analysis (divide into 6 zones)
    zones = _extract_zonal_intensities(img_array)
    
    # Peripheral vs Central analysis
    central_intensity, peripheral_intensity = _extract_peripheral_central(img_array)
    
    # Derived metrics
    apical_score = _calculate_apical_predominance(zones)
    peripheral_score = _calculate_peripheral_predominance(central_intensity, peripheral_intensity)
    symmetry_score = _calculate_bilateral_symmetry(zones)
    
    # Texture analysis
    entropy, homogeneity = _calculate_texture_metrics(img_array)
    
    return CXRFeatures(
        mean_intensity=mean_intensity,
        opacity_score=opacity_score,
        upper_left_intensity=zones['upper_left'],
        upper_right_intensity=zones['upper_right'],
        mid_left_intensity=zones['mid_left'],
        mid_right_intensity=zones['mid_right'],
        lower_left_intensity=zones['lower_left'],
        lower_right_intensity=zones['lower_right'],
        apical_predominance_score=apical_score,
        peripheral_predominance_score=peripheral_score,
        bilateral_symmetry_score=symmetry_score,
        texture_entropy=entropy,
        texture_homogeneity=homogeneity,
        central_intensity=central_intensity,
        peripheral_intensity=peripheral_intensity,
    )


def _calculate_opacity_score(img_array: np.ndarray) -> float:
    """Calculate how opaque (white) the image is.
    
    Higher score = more white = more pathology.
    """
    # Focus on lung region (assume center 60% of image)
    h, w = img_array.shape
    lung_region = img_array[
        int(h * 0.2):int(h * 0.8),
        int(w * 0.2):int(w * 0.8)
    ]
    
    # Opacity is inverse of blackness (lungs should be dark/black normally)
    # Higher values = more white = increased opacity
    return float(np.mean(lung_region))


def _extract_zonal_intensities(img_array: np.ndarray) -> Dict[str, float]:
    """Divide lung field into 6 zones and extract mean intensity.
    
    Zones: Upper/Mid/Lower x Left/Right
    """
    h, w = img_array.shape
    
    # Assume lung region occupies central 80% horizontally, 70% vertically
    top = int(h * 0.15)
    bottom = int(h * 0.85)
    left = int(w * 0.1)
    right = int(w * 0.9)
    
    lung_height = bottom - top
    lung_width = right - left
    
    # Divide into thirds vertically
    upper_boundary = top + lung_height // 3
    mid_boundary = top + 2 * lung_height // 3
    
    # Divide into halves horizontally
    center_x = left + lung_width // 2
    
    zones = {
        'upper_left': float(np.mean(img_array[top:upper_boundary, left:center_x])),
        'upper_right': float(np.mean(img_array[top:upper_boundary, center_x:right])),
        'mid_left': float(np.mean(img_array[upper_boundary:mid_boundary, left:center_x])),
        'mid_right': float(np.mean(img_array[upper_boundary:mid_boundary, center_x:right])),
        'lower_left': float(np.mean(img_array[mid_boundary:bottom, left:center_x])),
        'lower_right': float(np.mean(img_array[mid_boundary:bottom, center_x:right])),
    }
    
    return zones


def _extract_peripheral_central(img_array: np.ndarray) -> Tuple[float, float]:
    """Extract central vs peripheral intensity.
    
    Returns:
        (central_intensity, peripheral_intensity)
    """
    h, w = img_array.shape
    
    # Central region: inner 40% of lung field
    top = int(h * 0.15)
    bottom = int(h * 0.85)
    left = int(w * 0.1)
    right = int(w * 0.9)
    
    lung_height = bottom - top
    lung_width = right - left
    
    # Central region
    central_top = top + int(lung_height * 0.3)
    central_bottom = bottom - int(lung_height * 0.3)
    central_left = left + int(lung_width * 0.3)
    central_right = right - int(lung_width * 0.3)
    
    central = img_array[central_top:central_bottom, central_left:central_right]
    
    # Peripheral: outer ring
    peripheral_mask = np.ones_like(img_array, dtype=bool)
    peripheral_mask[central_top:central_bottom, central_left:central_right] = False
    peripheral_mask[:top, :] = False
    peripheral_mask[bottom:, :] = False
    peripheral_mask[:, :left] = False
    peripheral_mask[:, right:] = False
    
    peripheral = img_array[peripheral_mask]
    
    return float(np.mean(central)), float(np.mean(peripheral))


def _calculate_apical_predominance(zones: Dict[str, float]) -> float:
    """Calculate if pathology is concentrated in upper lobes (apical).
    
    Score > 1.0 = upper predominance (e.g., TB)
    Score < 1.0 = lower predominance
    Score ~ 1.0 = uniform
    """
    upper = (zones['upper_left'] + zones['upper_right']) / 2
    lower = (zones['lower_left'] + zones['lower_right']) / 2
    
    if lower < 0.01:  # Avoid division by zero
        return 1.0
    
    return float(upper / lower)


def _calculate_peripheral_predominance(central: float, peripheral: float) -> float:
    """Calculate if pathology is peripheral (e.g., COVID-19) vs central (e.g., Edema).
    
    Score > 1.0 = peripheral predominance (COVID-19)
    Score < 1.0 = central predominance (Edema)
    """
    if central < 0.01:
        return 1.0
    
    return float(peripheral / central)


def _calculate_bilateral_symmetry(zones: Dict[str, float]) -> float:
    """Calculate left-right symmetry.
    
    Score close to 1.0 = symmetric (bilateral disease)
    Score far from 1.0 = asymmetric (unilateral, focal lesion)
    """
    left_avg = (zones['upper_left'] + zones['mid_left'] + zones['lower_left']) / 3
    right_avg = (zones['upper_right'] + zones['mid_right'] + zones['lower_right']) / 3
    
    if right_avg < 0.01:
        return 1.0
    
    ratio = left_avg / right_avg
    # Return deviation from 1.0 (perfect symmetry)
    # Score close to 1.0 = symmetric
    return float(min(ratio, 1.0 / ratio))


def _calculate_texture_metrics(img_array: np.ndarray) -> Tuple[float, float]:
    """Calculate texture entropy and homogeneity.
    
    Returns:
        (entropy, homogeneity)
        
    Entropy: high = complex/irregular (GGO, infiltrates)
    Homogeneity: high = uniform (consolidation, normal)
    """
    # Focus on lung region
    h, w = img_array.shape
    lung_region = img_array[
        int(h * 0.2):int(h * 0.8),
        int(w * 0.2):int(w * 0.8)
    ]
    
    # Calculate histogram
    hist, _ = np.histogram(lung_region.flatten(), bins=32, range=(0, 1))
    hist = hist / (hist.sum() + 1e-8)  # Normalize
    
    # Entropy: measure of randomness
    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    entropy_normalized = float(entropy / np.log2(32))  # Normalize to [0, 1]
    
    # Homogeneity: inverse of variance (simple proxy)
    homogeneity = float(1.0 / (1.0 + np.var(lung_region)))
    
    return entropy_normalized, homogeneity

