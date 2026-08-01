from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.core.types import ExpertStatus
from typing import Any, Mapping

import numpy as np

from src.pipeline.noise.noise_generator import NoisyVolume
from src.pipeline.routing.adaptive_router import ExpertExecution


@dataclass(frozen=True)
class ExpertResult:
    """
    Immutable representation of an individual expert's denoising result.
    """
    ExpertName: str
    InputVolume: NoisyVolume
    OutputVolume: np.ndarray
    Confidence: float
    Weight: float
    ExecutionTime: float
    Status: ExpertStatus
    Statistics: Mapping[str, float]
    Metadata: Mapping[str, Any]
    LayerVersion: str


class BaseExpert(ABC):
    """
    Abstract interface for all denoising experts in the CT Image Noise Remover project.
    """
    
    @abstractmethod
    def execute(self, volume: NoisyVolume, execution: ExpertExecution) -> ExpertResult:
        """
        Executes the expert denoising algorithm.
        
        Args:
            volume: The NoisyVolume object containing the noisy data.
            execution: The execution plan configuration for this expert.
            
        Returns:
            ExpertResult: Immutable dataclass containing the denoised result.
        """
        pass
