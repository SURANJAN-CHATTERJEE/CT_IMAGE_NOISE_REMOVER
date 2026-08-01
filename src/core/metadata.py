from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Any, Tuple

@dataclass(frozen=True)
class DatasetFingerprint:
    DatasetHash: str
    PatientCount: int
    StudyCount: int
    SeriesCount: int
    SliceCount: int
    NoiseTypes: Tuple[str, ...]
    CreationTimestamp: str
    DatasetVersion: str

@dataclass(frozen=True)
class BaseMetadata:
    ProjectName: str = "CT_IMAGE_NOISE_REMOVER"
    ProjectVersion: str = "1.0.1"
    ArchitectureVersion: str = "2.0"
    ExperimentID: str = ""
    DatasetName: str = ""
    DatasetVersion: str = ""
    RandomSeed: int = 42
    GitCommit: str = ""
    Timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    HardwareInfo: Mapping[str, Any] = field(default_factory=dict)
    PythonVersion: str = "3.13"
    FrameworkVersion: str = "1.0.1"
    CUDAVersion: str = ""
    OS: str = ""
