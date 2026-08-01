from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.interfaces.fusion import FusionResult


@dataclass(frozen=True)
class ExplanationResult:
    """
    Immutable representation of the generated explainability data.
    """
    Parent: FusionResult
    ModelExplanations: Mapping[str, Mapping[str, Any]]
    RoutingExplanation: Mapping[str, Any]
    FusionExplanation: Mapping[str, Any]
    AttentionMaps: Mapping[str, np.ndarray]
    ContributionMaps: Mapping[str, np.ndarray]
    Metadata: Mapping[str, Any]
    ProcessingTime: float
    LayerVersion: str


class BaseExplainer(ABC):
    """
    Abstract interface for all explanation strategies in the CT Image Noise Remover project.
    """
    
    @abstractmethod
    def explain(self, fusion_result: FusionResult) -> ExplanationResult:
        """
        Generates comprehensive explainability data for the fusion result.
        
        Args:
            fusion_result: The FusionResult object from Layer 10.
            
        Returns:
            ExplanationResult: Immutable dataclass containing explainability data.
        """
        pass
