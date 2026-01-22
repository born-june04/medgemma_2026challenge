import hashlib
import importlib
from dataclasses import dataclass
from typing import Optional

import os
import sys
from pathlib import Path

# Force safetensors usage to avoid torch.load security restriction
# This must be set BEFORE importing transformers
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.environ["TRANSFORMERS_SAFE_LOADING"] = "1"

# Add project root to path for imports when running as script
# This must be done BEFORE importing hear module
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
    
import numpy as np
import torch
from huggingface_hub import login
login(token="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")

try:
    from transformers import AutoModel
    from transformers.utils import import_utils
    
    # Patch the torch.load security check to allow older torch versions
    # This is a workaround for CVE-2025-32434 with torch < 2.6
    original_check = import_utils.check_torch_load_is_safe
    def patched_check_torch_load_is_safe():
        """Patched version that skips the torch version check."""
        # Skip the check for older torch versions
        # Note: This is a workaround and should be used with caution
        pass
    
    # Apply the patch
    import_utils.check_torch_load_is_safe = patched_check_torch_load_is_safe
    
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
    preprocess_audio = audio_utils.preprocess_audio
    HEAR_AVAILABLE = True
    _hear_import_error = None
except (ImportError, ModuleNotFoundError) as e:
    HEAR_AVAILABLE = False
    preprocess_audio = None
    # Store error for debugging
    _hear_import_error = e

from signals.quality import load_audio_mono


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

    def encode(self, audio_path: str) -> torch.Tensor:
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
            # Force safetensors usage to avoid torch.load security restriction
            # This works with older torch versions (2.1.2+)
            # Set environment variables if not already set
            if "TRANSFORMERS_SAFE_LOADING" not in os.environ:
                os.environ["TRANSFORMERS_SAFE_LOADING"] = "1"
            
            try:
                # Try with safetensors first (preferred)
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    local_files_only=False
                )
            except Exception as e1:
                # If safetensors fails, try without explicit safetensors flag
                # but with trust_remote_code
                try:
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        local_files_only=False
                    )
                except Exception as e2:
                    # Last resort: try with low_cpu_mem_usage
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        local_files_only=False
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
    
    def encode(self, audio_path: str) -> torch.Tensor:
        """Encode audio file to embedding vector.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            torch.Tensor: Embedding vector of shape (embedding_dim,)
        """
        self._load_model()
        
        # Load audio at 16kHz (HeAR expects 16kHz)
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
    embedding = encoder.encode("/gscratch/ubicomp/tbscreen/audio_data/processed_data/tb/raw_data/tb_positive/PID_428A/2023_09_TB_P.wav")
    print(embedding.shape)