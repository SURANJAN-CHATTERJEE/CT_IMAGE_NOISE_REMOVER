from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from src.interfaces.fusion import FusionResult


@dataclass(frozen=True)
class EvaluationReport:
    """
    Immutable representation of the final evaluation and benchmarking results.
    """
    Parent: FusionResult
    ImageMetrics: Mapping[str, float]
    PipelineMetrics: Mapping[str, Any]
    BenchmarkMetrics: Mapping[str, Any]
    Summary: Mapping[str, Any]
    Metadata: Mapping[str, Any]
    ProcessingTime: float
    LayerVersion: str


class BaseEvaluator(ABC):
    """
    Abstract interface for all evaluation strategies in the CT Image Noise Remover project.
    """
    
    @abstractmethod
    def evaluate(self, fusion_result: FusionResult, ground_truth: Optional[np.ndarray] = None) -> EvaluationReport:
        """
        Evaluates the complete framework, producing image, pipeline, and benchmark metrics.
        
        Args:
            fusion_result: The FusionResult object from Layer 10.
            ground_truth: Optional ground truth volume for training mode evaluation.
            
        Returns:
            EvaluationReport: Immutable dataclass containing comprehensive evaluation data.
        """
        pass
