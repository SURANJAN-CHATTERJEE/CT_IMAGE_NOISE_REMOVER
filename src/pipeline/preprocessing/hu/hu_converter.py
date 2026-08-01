import copy
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Any, Tuple

import numpy as np

from src.pipeline.preprocessing.dicom.input_manager import InputVolume

logger = logging.getLogger(__name__)


class HUConversionError(Exception):
    """Base exception for HU conversion errors."""
    pass


class InvalidVolumeError(HUConversionError):
    """Raised when the input volume is invalid (e.g., empty, wrong type)."""
    pass


class MissingCalibrationError(HUConversionError):
    """Raised when essential calibration data is completely missing."""
    pass


class InvalidCalibrationError(HUConversionError):
    """Raised when calibration values (slope, intercept) are invalid."""
    pass


@dataclass(frozen=True)
class HUVolume:
    """
    Immutable representation of a Hounsfield Unit (HU) calibrated CT volume.
    """
    Volume: np.ndarray
    Metadata: Mapping[str, Any]
    Statistics: Mapping[str, float]
    MinHU: float
    MaxHU: float
    MeanHU: float
    StdHU: float
    VolumeShape: Tuple[int, ...]
    VolumeDtype: str
    EstimatedMemoryMB: float
    ProcessingTime: float
    LayerVersion: str


class HUConverter:
    """
    Engine for converting raw CT voxel values into calibrated Hounsfield Units (HU).
    This is the exclusive layer responsible for HU calibration.
    """
    
    VERSION: str = "2.0.0"
    
    def convert(
        self, 
        input_volume: Any, 
        clip: bool = False, 
        hu_min: float = -1024.0, 
        hu_max: float = 3071.0
    ) -> HUVolume:
        """
        Converts the raw CT volume into Hounsfield Units.
        
        Args:
            input_volume: The InputVolume object from Layer 1 containing raw volume and metadata.
            clip: Whether to clip the resulting HU values.
            hu_min: The minimum HU value if clipping is enabled.
            hu_max: The maximum HU value if clipping is enabled.
            
        Returns:
            HUVolume: An immutable dataclass containing the HU calibrated volume and metadata.
            
        Raises:
            InvalidVolumeError: If the input volume is missing or malformed.
            MissingCalibrationError: If metadata dictionary is missing or inaccessible.
            InvalidCalibrationError: If slope, intercept, or clipping arguments are invalid.
            HUConversionError: If integrity verification fails.
        """
        start_time = time.time()
        
        if clip and hu_min >= hu_max:
            raise InvalidCalibrationError(f"Invalid clipping range: hu_min ({hu_min}) >= hu_max ({hu_max}).")
            
        self._validate_input_volume(input_volume)
        
        raw_volume: np.ndarray = input_volume.Volume
        
        metadata: dict[str, Any] = copy.deepcopy(dict(input_volume.Metadata))
        
        slope, intercept = self._extract_calibration(metadata)
        
        patient_id = metadata.get("PatientID", "UNKNOWN")
        series_uid = getattr(input_volume, "SeriesInstanceUID", metadata.get("SeriesInstanceUID", "UNKNOWN"))
        
        logger.info(f"PatientID: {patient_id}")
        logger.info(f"SeriesInstanceUID: {series_uid}")
        logger.info(f"Input Shape: {raw_volume.shape}")
        logger.info(f"Input Dtype: {raw_volume.dtype}")
        logger.info(f"Slope: {slope}, Intercept: {intercept}")
        
        hu_array = raw_volume.astype(np.float32)
        
        if slope != 1.0:
            hu_array *= np.float32(slope)
            
        if intercept != 0.0:
            hu_array += np.float32(intercept)
            
        if clip:
            logger.debug(f"Clipping HU values to range [{hu_min}, {hu_max}]")
            hu_array = np.clip(hu_array, hu_min, hu_max)
            
        if np.isnan(hu_array).any():
            raise HUConversionError("Converted volume contains NaN values.")
        if np.isinf(hu_array).any():
            raise HUConversionError("Converted volume contains Inf values.")
            
        hu_array.flags.writeable = False
        
        vol_shape = hu_array.shape
        vol_dtype = str(hu_array.dtype)
        est_mem_mb = float(hu_array.nbytes / (1024 * 1024))
        
        if vol_shape != raw_volume.shape:
            raise HUConversionError(f"Shape inconsistency detected: {vol_shape} != {raw_volume.shape}")
        if vol_dtype != "float32":
            raise HUConversionError(f"Dtype inconsistency detected: {vol_dtype} != float32")
        expected_bytes = np.prod(vol_shape) * 4
        if abs(est_mem_mb - (expected_bytes / (1024 * 1024))) > 1e-6:
            raise HUConversionError("Memory estimation inconsistency detected.")
            
        statistics = self._compute_statistics(hu_array)
        
        processing_time = time.time() - start_time
        
        metadata["ConversionTimestamp"] = datetime.now().isoformat()
        metadata["LayerVersion"] = self.LayerVersion
        metadata["ProcessingTime"] = processing_time
        
        frozen_metadata = MappingProxyType(metadata)
        frozen_statistics = MappingProxyType(statistics)
        
        logger.info(f"Output Shape: {vol_shape}")
        logger.info(f"Output Dtype: {vol_dtype}")
        logger.info(f"Minimum HU: {statistics['Minimum']:.2f}")
        logger.info(f"Maximum HU: {statistics['Maximum']:.2f}")
        logger.info(f"Mean HU: {statistics['Mean']:.2f}")
        logger.info(f"Processing Time: {processing_time:.4f} s")
        
        return HUVolume(
            Volume=hu_array,
            Metadata=frozen_metadata,
            Statistics=frozen_statistics,
            MinHU=statistics["Minimum"],
            MaxHU=statistics["Maximum"],
            MeanHU=statistics["Mean"],
            StdHU=statistics["StandardDeviation"],
            VolumeShape=vol_shape,
            VolumeDtype=vol_dtype,
            EstimatedMemoryMB=est_mem_mb,
            ProcessingTime=processing_time,
            LayerVersion=self.LayerVersion
        )
        
    def _validate_input_volume(self, input_volume: Any) -> None:
        """
        Validates the structure and data types of the input volume.
        """
        if not isinstance(input_volume, InputVolume):
            raise InvalidVolumeError(f"Input object must be an InputVolume, got {type(input_volume).__name__}.")
            
        if not hasattr(input_volume, "Volume") or not hasattr(input_volume, "Metadata"):
            raise InvalidVolumeError("Input object lacks 'Volume' or 'Metadata' attributes.")
            
        raw_volume = input_volume.Volume
        
        if not isinstance(raw_volume, np.ndarray):
            raise InvalidVolumeError(f"Volume must be a NumPy array, got {type(raw_volume)}")
            
        if raw_volume.size == 0:
            raise InvalidVolumeError("Input volume is empty.")
            
        if raw_volume.ndim < 2:
            raise InvalidVolumeError(f"Invalid volume dimensions: {raw_volume.ndim}. Expected at least 2D.")
            
        valid_dtypes = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.float32, np.float64]
        if raw_volume.dtype.type not in valid_dtypes:
            raise InvalidVolumeError(f"Invalid volume dtype: {raw_volume.dtype}. Must be a supported numeric type.")

    def _extract_calibration(self, metadata: dict[str, Any]) -> Tuple[float, float]:
        """
        Safely extracts RescaleSlope and RescaleIntercept from metadata.
        """
        if not metadata:
            raise MissingCalibrationError("Metadata dictionary is missing.")
            
        try:
            slope = float(metadata.get("RescaleSlope", 1.0))
            intercept = float(metadata.get("RescaleIntercept", 0.0))
        except (ValueError, TypeError):
            raise InvalidCalibrationError("RescaleSlope or RescaleIntercept must be numeric values.")
            
        if np.isnan(slope) or np.isinf(slope):
            raise InvalidCalibrationError(f"Invalid RescaleSlope: {slope}")
            
        if np.isnan(intercept) or np.isinf(intercept):
            raise InvalidCalibrationError(f"Invalid RescaleIntercept: {intercept}")
            
        if abs(slope) < 1e-6:
            raise InvalidCalibrationError(f"RescaleSlope is too close to zero: {slope}")
            
        return slope, intercept

    def _compute_statistics(self, volume: np.ndarray) -> dict[str, float]:
        """
        Automatically computes key statistical metrics of the HU volume.
        """
        min_hu = float(np.min(volume))
        max_hu = float(np.max(volume))
        return {
            "Minimum": min_hu,
            "Maximum": max_hu,
            "Mean": float(np.mean(volume)),
            "Median": float(np.median(volume)),
            "StandardDeviation": float(np.std(volume)),
            "Variance": float(np.var(volume)),
            "DynamicRange": max_hu - min_hu
        }
