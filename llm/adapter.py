from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMAdapter(ABC):
    """Adapter interface for clinician-facing generation.

    Clinical intent: produce conservative summaries without diagnosis.
    """

    @abstractmethod
    def generate(self, structured_input: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class MedGemmaAdapter(LLMAdapter):
    """Placeholder for Med-Gemma integration."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        raise NotImplementedError("Med-Gemma loading is not implemented in this demo.")

    def generate(self, structured_input: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
