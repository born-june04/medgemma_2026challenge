import hashlib
import importlib
from dataclasses import dataclass
from typing import Optional, Union

import os
import sys
from pathlib import Path

# Set LD_LIBRARY_PATH to use conda environment's libstdc++ (fixes GLIBCXX version issues)
# This must be done before importing any modules that depend on C++ libraries
if "CONDA_PREFIX" in os.environ:
    conda_lib = os.path.join(os.environ["CONDA_PREFIX"], "lib")
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if conda_lib not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{current_ld_path}" if current_ld_path else conda_lib

# Add project root to path for imports when running as script
# This must be done BEFORE importing hear module
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
    
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from huggingface_hub import login
login(token="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")

TRANSFORMERS_AVAILABLE = True
import transformers
from transformers import AutoModel
import transformers.utils.import_utils as import_utils
import_utils._torch_version = "2.6.0"

audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
preprocess_audio = audio_utils.preprocess_audio
HEAR_AVAILABLE = True
_hear_import_error = None
from Agent1_Audio.signals.quality import load_audio_mono


@dataclass
class EncoderMetadata:
    name: str
    is_stub: bool
    embedding_dim: int


class AudioEncoderBase:
    """Base interface for audio encoders.

    Clinical intent: provide a stable, comparable embedding for downstream signal scoring.
    """

    def __init__(self, metadata: EncoderMetadata):
        self.metadata = metadata

    def encode(self, audio_path: Union[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class StubAudioEncoder(AudioEncoderBase):
    """Deterministic stub encoder used when checkpoints are unavailable."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__(EncoderMetadata(name="stub_hear", is_stub=True, embedding_dim=embedding_dim))

    def encode(self, audio_path: str) -> torch.Tensor:
        digest = _hash_file(audio_path)
        rng = np.random.RandomState(int(digest[:8], 16))
        embedding = rng.normal(0.0, 1.0, size=self.metadata.embedding_dim).astype(np.float32)
        return torch.from_numpy(embedding)


class HeARAudioEncoder(AudioEncoderBase):
    """Google HeAR (HAI-DEF) encoder.

    Clinical intent: map audio into a representation that captures salient acoustic patterns.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, embedding_dim: Optional[int] = None):
        # Use checkpoint_path as model name, or default to the official HeAR model
        self.model_name = checkpoint_path if checkpoint_path else "google/hear-pytorch"
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for HeARAudioEncoder. "
                "Install it with: pip install transformers==4.50.3"
            )
        
        if not HEAR_AVAILABLE:
            error_msg = (
                "hear package is required for HeARAudioEncoder. "
                "Install it with: git clone https://github.com/Google-Health/hear.git"
            )
            if _hear_import_error is not None:
                error_msg += f"\nImport error: {_hear_import_error}"
            raise ImportError(error_msg)
        
        # Initialize with placeholder embedding_dim, will be updated after model loads
        super().__init__(EncoderMetadata(name="hear_hai_def", is_stub=False, embedding_dim=embedding_dim or 512))
        self._user_embedding_dim = embedding_dim
        self.model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._embedding_dim_initialized = False
    
    def _load_model(self):
        """Lazy load the model only when needed."""
        if self.model is None:
            print(f"Loading HeAR model (transformers=={transformers.__version__})...")
            
            # Try multiple loading strategies
            load_attempts = [
                {
                    "name": "Primary (with torch_dtype)",
                    "kwargs": {
                        "trust_remote_code": True,
                        "local_files_only": False,
                        "torch_dtype": torch.float32,
                    }
                },
                {
                    "name": "Fallback 1 (ignore_mismatched_sizes)",
                    "kwargs": {
                        "trust_remote_code": True,
                        "local_files_only": False,
                        "ignore_mismatched_sizes": True,
                    }
                },
                {
                    "name": "Fallback 2 (ignore_mismatched_sizes + use_safetensors=False)",
                    "kwargs": {
                        "trust_remote_code": True,
                        "use_safetensors": False,
                        "ignore_mismatched_sizes": True,
                    }
                },
            ]
            
            last_error = None
            for attempt in load_attempts:
                try:
                    print(f"Attempting {attempt['name']}...")
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        **attempt["kwargs"]
                    )
                    print(f"Successfully loaded model using {attempt['name']}")
                    break
                except Exception as e:
                    last_error = e
                    print(f"{attempt['name']} failed: {str(e)[:200]}")
                    continue
            
            if self.model is None:
                raise RuntimeError(
                    f"Failed to load model after {len(load_attempts)} attempts. "
                    f"Last error: {last_error}"
                )
                
            self.model.to(self._device)
            self.model.eval()
            
            # Update embedding_dim from model config if not explicitly set
            if not self._embedding_dim_initialized:
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                    actual_dim = self.model.config.hidden_size
                else:
                    # Default to 512 if we can't determine it
                    actual_dim = 512
                
                if self._user_embedding_dim is None:
                    self.metadata.embedding_dim = actual_dim
                self._embedding_dim_initialized = True
    
    def encode(self, audio_path: Union[str, torch.Tensor]) -> torch.Tensor:
        """Encode audio file, spectrogram, or scalogram PNG to embedding vector.
        
        Args:
            audio_path: Either:
                - Path to audio file (str): Will load and preprocess the audio
                - Path to scalogram PNG file (str ending with .png): Will load and convert PNG to spectrogram
                - Preprocessed spectrogram tensor (torch.Tensor): Shape [batch_size, 1, 192, 128]
                  or [1, 192, 128]. If spectrogram is provided, skips audio loading/preprocessing.
            
        Returns:
            torch.Tensor: Embedding vector of shape (embedding_dim,)
        """
        self._load_model()
        
        # Check if input is a spectrogram tensor or file path
        if isinstance(audio_path, torch.Tensor):
            # Input is already a spectrogram
            spectrogram_batch = audio_path
            # Ensure it has batch dimension
            if spectrogram_batch.ndim == 3:
                spectrogram_batch = spectrogram_batch.unsqueeze(0)  # (1, 1, 192, 128)
            elif spectrogram_batch.ndim == 4:
                pass  # Already has batch dimension
            else:
                raise ValueError(
                    f"Spectrogram must have 3 or 4 dimensions, got {spectrogram_batch.ndim}. "
                    f"Expected shape: [1, 192, 128] or [batch_size, 1, 192, 128]"
                )
        elif isinstance(audio_path, str) and audio_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Input is a scalogram PNG/image file
            img = Image.open(audio_path).convert('L')  # Convert to grayscale
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize to [0, 1] range
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
            
            # Convert to tensor and add channel dimension: (H, W) -> (1, H, W)
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)
            
            # Resize to HeAR expected size (192, 128) using bilinear interpolation
            # HeAR expects [batch, channel, height, width] = [1, 1, 192, 128]
            spectrogram_batch = F.interpolate(
                img_tensor.unsqueeze(0),  # Add batch dim: (1, 1, H, W)
                size=(192, 128),
                mode='bilinear',
                align_corners=False
            )  # (1, 1, 192, 128)
        else:
            # Input is audio path - load and preprocess
            audio, sr = load_audio_mono(audio_path, target_sr=16000)
            
            # Convert to torch tensor and add batch dimension
            # HeAR expects shape (batch_size, num_samples)
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, num_samples)
            
            # Preprocess audio to spectrogram
            with torch.no_grad():
                spectrogram_batch = preprocess_audio(audio_tensor)
        
        # Move to device if model is on GPU
        if self._device.type == "cuda":
            spectrogram_batch = spectrogram_batch.to(self._device)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model.forward(
                spectrogram_batch,
                return_dict=True,
                output_hidden_states=True
            )
            
            # Extract embedding from the last hidden state
            # The output shape is typically (batch_size, seq_len, hidden_dim)
            # We take the mean pooling over the sequence dimension
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze(0)  # (hidden_dim,)
            
            # Move back to CPU if needed
            if self._device.type == "cuda":
                embedding = embedding.cpu()
        
        return embedding


def get_audio_encoder(use_stub: bool, checkpoint_path: Optional[str]) -> AudioEncoderBase:
    if use_stub or not checkpoint_path:
        return StubAudioEncoder()
    try:
        return HeARAudioEncoder(checkpoint_path=checkpoint_path)
    except (NotImplementedError, ImportError) as e:
        # Fall back to stub encoder if HeAR dependencies are not available
        return StubAudioEncoder()


def _hash_file(path: str, max_bytes: int = 4_000_000) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        chunk = handle.read(max_bytes)
        hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    encoder = HeARAudioEncoder()
    # Test with scalogram PNG file
    png_path = "/gscratch/scrubbed/june0604/medgemma_2026challenge/data/Chest_Diseases_Dataset/1. COVID-19/CSI/Image 01 (14).png"
    embedding = encoder.encode(png_path)
    print(f"Embedding shape: {embedding.shape}")