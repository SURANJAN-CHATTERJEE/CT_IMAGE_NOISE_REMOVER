from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from src.pipeline.restoration.expert_execution_manager import ExecutionResult
@dataclass(frozen=True)
class FusionContext:
    TrainingMode: bool
    GroundTruth: Optional[np.ndarray]
    DatasetName: str
    ExperimentID: str
    Epoch: int
    Checkpoint: str
    EvaluationMode: bool



@dataclass(frozen=True)
class FusionResult:
    """
    Immutable representation of the final fused CT volume.
    """
    Parent: ExecutionResult
    FinalVolume: np.ndarray
    FusionStrategy: str
    ExpertContributions: Mapping[str, float]
    FusionWeights: Mapping[str, float]
    QualityMetrics: Mapping[str, float]
    Metadata: Mapping[str, Any]
    ProcessingTime: float
    LayerVersion: str


class BaseFusionStrategy(ABC):
    """
    Abstract interface for all fusion strategies in the CT Image Noise Remover project.
    """
    
    @abstractmethod
    def fuse(
        self, 
        execution_result: ExecutionResult, context: Optional[FusionContext] = None,
        ground_truth: Optional[np.ndarray] = None
    ) -> FusionResult:
        """
        Fuses multiple expert outputs into one final denoised CT volume.
        
        Args:
            execution_result: The ExecutionResult containing individual expert outputs.
            ground_truth: Optional ground truth volume for training mode evaluation.
            
        Returns:
            FusionResult: Immutable dataclass containing the fused volume and metadata.
        """
        pass


class ReliabilityProvider(ABC):
    """
    Abstract interface for providing expert reliability scores.
    """
    
    @abstractmethod
    def get_reliability(self, expert_name: str, execution_result: ExecutionResult) -> float:
        """
        Retrieves or computes the reliability score for an expert.
        """
        pass
