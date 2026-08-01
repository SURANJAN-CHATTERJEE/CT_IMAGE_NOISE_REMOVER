import copy
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Any, Tuple, Optional, Dict

import numpy as np

from src.pipeline.preprocessing.hu.hu_converter import HUVolume

logger = logging.getLogger(__name__)


class WindowingError(Exception):
    """Base exception for windowing errors."""
    pass


class InvalidWindowError(WindowingError):
    """Raised when window parameters (center, width) are invalid."""
    pass


class InvalidPresetError(WindowingError):
    """Raised when a requested windowing preset is not recognized."""
    pass


@dataclass(frozen=True)
class WindowedVolume:
    """
    Immutable representation of a windowed and normalized CT volume.
    """
    OriginalHUVolume: HUVolume
    Volume: np.ndarray
    Metadata: Mapping[str, Any]
    Statistics: Mapping[str, float]
    WindowCenter: float
    WindowWidth: float
    Preset: str
    MinValue: float
    MaxValue: float
    MeanValue: float
    StdValue: float
    VolumeShape: Tuple[int, ...]
    VolumeDtype: str
    EstimatedMemoryMB: float
    ProcessingTime: float
    LayerVersion: str


class CTWindowing:
    """
    Engine for converting HU calibrated CT volumes into windowed 0-1 scaled arrays.
    This is the exclusive layer responsible for CT windowing.
    """
    
    VERSION: str = "1.0.1"
    
    PRESETS: Dict[str, Tuple[float, float]] = {
        "lung": (-600.0, 1500.0),
        "soft_tissue": (40.0, 400.0),
        "bone": (400.0, 1500.0),
        "brain": (40.0, 80.0),
        "abdomen": (40.0, 400.0),
        "liver": (30.0, 150.0),
        "mediastinum": (50.0, 350.0),
        "angio": (300.0, 600.0)
    }

    def apply(
        self,
        hu_volume: Any,
        preset: Optional[str] = None,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None
    ) -> WindowedVolume:
        """
        Applies CT windowing to the HU volume.
        
        Args:
            hu_volume: The HUVolume object from Layer 2.
            preset: The name of a standard CT windowing preset.
            window_center: Custom window center (if preset is not used).
            window_width: Custom window width (if preset is not used).
            
        Returns:
            WindowedVolume: An immutable dataclass containing the windowed volume.
            
        Raises:
            WindowingError: For general windowing failures or integrity issues.
            InvalidPresetError: If the preset name is unknown.
            InvalidWindowError: If window parameters are missing or mathematically invalid.
        """
        start_time = time.time()
        
        self._validate_input_volume(hu_volume)
        
        center, width, preset_name = self._resolve_parameters(preset, window_center, window_width)
        
        metadata: dict[str, Any] = copy.deepcopy(dict(hu_volume.Metadata))
        raw_volume: np.ndarray = hu_volume.Volume
        
        patient_id = metadata.get("PatientID", "UNKNOWN")
        series_uid = metadata.get("SeriesInstanceUID", "UNKNOWN")
        
        logger.info(f"PatientID: {patient_id}")
        logger.info(f"SeriesUID: {series_uid}")
        logger.info(f"Input Shape: {raw_volume.shape}")
        
        lower_bound = float(center - (width / 2.0))
        upper_bound = float(center + (width / 2.0))
        
        logger.debug(f"Applying windowing: Center={center}, Width={width}")
        logger.debug(f"HU Clip Range: [{lower_bound}, {upper_bound}]")
        
        arr = raw_volume.astype(np.float32)
        arr = np.clip(arr, lower_bound, upper_bound)
        arr = (arr - lower_bound) / width
        
        if np.isnan(arr).any():
            raise WindowingError("Resulting volume contains NaN values.")
        if np.isinf(arr).any():
            raise WindowingError("Resulting volume contains Inf values.")
            
        arr.flags.writeable = False
        
        vol_shape = arr.shape
        vol_dtype = str(arr.dtype)
        est_mem_mb = float(arr.nbytes / (1024 * 1024))
        
        if vol_shape != raw_volume.shape:
            raise WindowingError(f"Shape inconsistency detected: {vol_shape} != {raw_volume.shape}")
        if vol_dtype != "float32":
            raise WindowingError(f"Dtype inconsistency detected: {vol_dtype} != float32")
        expected_bytes = np.prod(vol_shape) * 4
        if abs(est_mem_mb - (expected_bytes / (1024 * 1024))) > 1e-6:
            raise WindowingError("Memory estimation inconsistency detected.")
            
        statistics = self._compute_statistics(arr)
        
        if statistics["Minimum"] < 0.0 or statistics["Maximum"] > 1.0:
            raise WindowingError(
                f"Normalization failed. Values out of [0, 1] range: "
                f"Min={statistics['Minimum']}, Max={statistics['Maximum']}"
            )
            
        processing_time = time.time() - start_time
        
        metadata["WindowingTimestamp"] = datetime.now().isoformat()
        metadata["WindowCenter"] = center
        metadata["WindowWidth"] = width
        metadata["WindowPreset"] = preset_name
        metadata["WindowRange"] = (lower_bound, upper_bound)
        metadata["LayerVersion"] = self.LayerVersion
        metadata["ProcessingTime"] = processing_time
        
        frozen_metadata = MappingProxyType(metadata)
        frozen_statistics = MappingProxyType(statistics)
        
        logger.info(f"Preset: {preset_name}")
        logger.info(f"Window Center: {center}")
        logger.info(f"Window Width: {width}")
        logger.info(f"Output Shape: {vol_shape}")
        logger.info(f"Output Dtype: {vol_dtype}")
        logger.info(f"Minimum: {statistics['Minimum']:.4f}")
        logger.info(f"Maximum: {statistics['Maximum']:.4f}")
        logger.info(f"Mean: {statistics['Mean']:.4f}")
        logger.info(f"Dynamic Range: {statistics['DynamicRange']:.4f}")
        logger.info(f"Processing Time: {processing_time:.4f} s")
        
        return WindowedVolume(
            OriginalHUVolume=hu_volume,
            Volume=arr,
            Metadata=frozen_metadata,
            Statistics=frozen_statistics,
            WindowCenter=center,
            WindowWidth=width,
            Preset=preset_name,
            MinValue=statistics["Minimum"],
            MaxValue=statistics["Maximum"],
            MeanValue=statistics["Mean"],
            StdValue=statistics["StandardDeviation"],
            VolumeShape=vol_shape,
            VolumeDtype=vol_dtype,
            EstimatedMemoryMB=est_mem_mb,
            ProcessingTime=processing_time,
            LayerVersion=self.LayerVersion
        )

    def _validate_input_volume(self, hu_volume: Any) -> None:
        """
        Validates the structure and data types of the input HU volume.
        """
        if not isinstance(hu_volume, HUVolume):
            raise WindowingError(f"Input object must be an HUVolume, got {type(hu_volume).__name__}.")
            
        if not hasattr(hu_volume, "Volume") or not hasattr(hu_volume, "Metadata"):
            raise WindowingError("Input object lacks 'Volume' or 'Metadata' attributes.")
            
        if not isinstance(hu_volume.Volume, np.ndarray):
            raise WindowingError(f"Volume must be a NumPy array, got {type(hu_volume.Volume)}")
            
        if hu_volume.Volume.size == 0:
            raise WindowingError("Input volume is empty.")
            
        if not isinstance(hu_volume.Metadata, Mapping) and not isinstance(hu_volume.Metadata, dict):
            raise WindowingError("Metadata must be a dictionary or MappingProxyType.")

    def _resolve_parameters(
        self, 
        preset: Optional[str], 
        center: Optional[float], 
        width: Optional[float]
    ) -> Tuple[float, float, str]:
        """
        Resolves window center and width from either a preset name or direct numerical inputs.
        """
        if preset is not None:
            preset_key = preset.lower()
            if preset_key not in self.PRESETS:
                raise InvalidPresetError(f"Preset '{preset}' is not supported.")
            c, w = self.PRESETS[preset_key]
            return c, w, preset_key
            
        if center is not None and width is not None:
            try:
                c = float(center)
                w = float(width)
            except (ValueError, TypeError):
                raise InvalidWindowError("Window center and width must be numeric values.")
                
            if np.isnan(c) or np.isinf(c) or np.isnan(w) or np.isinf(w):
                raise InvalidWindowError("Window center and width must be finite numbers.")
                
            if w <= 0:
                raise InvalidWindowError(f"Window width must be strictly positive, got {w}")
                
            return c, w, "custom"
            
        raise WindowingError("You must provide either a 'preset' or both 'window_center' and 'window_width'.")

    def _compute_statistics(self, volume: np.ndarray) -> dict[str, float]:
        """
        Automatically computes key statistical metrics of the windowed volume.
        """
        min_val = float(np.min(volume))
        max_val = float(np.max(volume))
        return {
            "Minimum": min_val,
            "Maximum": max_val,
            "Mean": float(np.mean(volume)),
            "Median": float(np.median(volume)),
            "StandardDeviation": float(np.std(volume)),
            "Variance": float(np.var(volume)),
            "DynamicRange": max_val - min_val
        }
