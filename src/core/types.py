from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Any, Tuple

class ExpertStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    TIMEOUT = "TIMEOUT"

@dataclass(frozen=True)
class NoiseOperation:
    Type: str
    Intensity: float
    ScaleFactor: float
    Timestamp: str
    Parameters: Mapping[str, Any]

@dataclass(frozen=True)
class ExpertExplanation:
    ExpertName: str
    Confidence: float
    InferenceTime: float
    ExplanationMethod: str
    GradCAM_SaliencyMean: float
    IntegratedGradients_Contribution: float
