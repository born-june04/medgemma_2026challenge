import numpy as np
import torch


def cxr_abnormality_score(embedding: np.ndarray) -> float:
    """Heuristic score for CXR embedding deviation.

    Clinical intent: surface potentially atypical imaging patterns without diagnosis.
    """
    embedding_np = _to_numpy(embedding)
    norm = float(np.linalg.norm(embedding_np))
    score = 1.0 / (1.0 + np.exp(-0.02 * (norm - 20.0)))
    return float(np.clip(score, 0.0, 1.0))


def _to_numpy(embedding):
    if isinstance(embedding, torch.Tensor):
        return embedding.detach().cpu().numpy()
    return np.asarray(embedding)
