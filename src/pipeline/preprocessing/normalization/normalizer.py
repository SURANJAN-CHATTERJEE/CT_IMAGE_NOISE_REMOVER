import copy
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Any, Tuple, Dict

import numpy as np

from src.pipeline.preprocessing.windowing.windowing import WindowedVolume

logger = logging.getLogger(__name__)


class NormalizationError(Exception):
    """Base exception for normalization errors."""
    pass


class InvalidNormalizationMethod(NormalizationError):
    """Raised when an unsupported normalization method is requested."""
    pass


class ConstantVolumeError(NormalizationError):
    """Raised when the volume has zero variance or zero scale range."""
    pass


@dataclass(frozen=True)
class NormalizedVolume:
    """
    Immutable representation of a deep-learning normalized CT volume.
    """
    Parent: WindowedVolume
    Volume: np.ndarray
    Metadata: Mapping[str, Any]
    Statistics: Mapping[str, float]
    NormalizationMethod: str
    NormalizationParameters: Mapping[str, Any]
    MinValue: float
    MaxValue: float
    MeanValue: float
    StdValue: float
    VolumeShape: Tuple[int, ...]
    VolumeDtype: str
    EstimatedMemoryMB: float
    ProcessingTime: float
    LayerVersion: str


class CTNormalizer:
    """
    Engine for applying deep learning normalization techniques to CT volumes.
    This is the exclusive layer responsible for normalization.
    """
    
    VERSION: str = "1.0.1"
    
    SUPPORTED_METHODS = {"minmax", "zscore", "robust", "percentile", "none"}

    def normalize(
        self,
        windowed_volume: Any,
        method: str = "minmax",
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0
    ) -> NormalizedVolume:
        """
        Applies mathematical normalization to the windowed CT volume.
        
        Args:
            windowed_volume: The WindowedVolume object from Layer 3.
            method: Normalization method (minmax, zscore, robust, percentile, none).
            lower_percentile: Lower percentile bound (used only if method='percentile').
            upper_percentile: Upper percentile bound (used only if method='percentile').
            
        Returns:
            NormalizedVolume: An immutable dataclass containing the normalized volume.
            
        Raises:
            NormalizationError: For general normalization failures or integrity issues.
            InvalidNormalizationMethod: If the specified method is unknown.
            ConstantVolumeError: If the volume lacks sufficient variance for scaling.
        """
        start_time = time.time()
        
        self._validate_input_volume(windowed_volume)
        
        method_key = method.lower()
        if method_key not in self.SUPPORTED_METHODS:
            raise InvalidNormalizationMethod(
                f"Unsupported normalization method: '{method}'. Supported: {self.SUPPORTED_METHODS}"
            )
            
        if method_key == "percentile":
            if lower_percentile < 0.0 or upper_percentile > 100.0 or lower_percentile >= upper_percentile:
                raise NormalizationError(f"Invalid percentiles: lower={lower_percentile}, upper={upper_percentile}")
                
        metadata: dict[str, Any] = copy.deepcopy(dict(windowed_volume.Metadata))
        raw_volume: np.ndarray = windowed_volume.Volume
        
        patient_id = metadata.get("PatientID", "UNKNOWN")
        series_uid = metadata.get("SeriesInstanceUID", "UNKNOWN")
        
        logger.info(f"PatientID: {patient_id}")
        logger.info(f"SeriesUID: {series_uid}")
        logger.info(f"Input Shape: {raw_volume.shape}")
        logger.info(f"Normalization Method: {method_key}")
        
        arr = raw_volume.astype(np.float32)
        norm_params: Dict[str, Any] = {"Method": method_key}
        
        if method_key == "minmax":
            min_val = np.min(arr)
            max_val = np.max(arr)
            if np.isclose(min_val, max_val):
                raise ConstantVolumeError("Volume has zero range (min == max). Cannot apply minmax normalization.")
            arr = (arr - min_val) / (max_val - min_val)
            norm_params.update({"Min": float(min_val), "Max": float(max_val)})
            
        elif method_key == "zscore":
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            if np.isclose(std_val, 0.0):
                raise ConstantVolumeError("Volume has zero variance. Cannot apply zscore normalization.")
            arr = (arr - mean_val) / std_val
            norm_params.update({"Mean": float(mean_val), "Std": float(std_val)})
            
        elif method_key == "robust":
            median_val = np.median(arr)
            p75, p25 = np.percentile(arr, [75.0, 25.0])
            iqr = p75 - p25
            if np.isclose(iqr, 0.0):
                raise ConstantVolumeError("Volume has zero IQR. Cannot apply robust normalization.")
            arr = (arr - median_val) / iqr
            norm_params.update({"Median": float(median_val), "IQR": float(iqr)})
            
        elif method_key == "percentile":
            p_low, p_high = np.percentile(arr, [lower_percentile, upper_percentile])
            if np.isclose(p_low, p_high):
                raise ConstantVolumeError(
                    f"Volume has zero range between percentiles {lower_percentile} and {upper_percentile}."
                )
            arr = (arr - p_low) / (p_high - p_low)
            arr = np.clip(arr, 0.0, 1.0)
            norm_params.update({
                "LowerPercentile": float(lower_percentile),
                "UpperPercentile": float(upper_percentile),
                "PLow": float(p_low),
                "PHigh": float(p_high)
            })
            
        elif method_key == "none":
            pass
            
        if arr.size == 0:
            raise NormalizationError("Resulting volume is empty.")
        if np.isnan(arr).any():
            raise NormalizationError("Resulting volume contains NaN values.")
        if np.isinf(arr).any():
            raise NormalizationError("Resulting volume contains Inf values.")
            
        arr.flags.writeable = False
        
        vol_shape = arr.shape
        vol_dtype = str(arr.dtype)
        est_mem_mb = float(arr.nbytes / (1024 * 1024))
        
        if vol_shape != raw_volume.shape:
            raise NormalizationError(f"Shape inconsistency detected: {vol_shape} != {raw_volume.shape}")
        if vol_dtype != "float32":
            raise NormalizationError(f"Dtype inconsistency detected: {vol_dtype} != float32")
        expected_bytes = np.prod(vol_shape) * 4
        if abs(est_mem_mb - (expected_bytes / (1024 * 1024))) > 1e-6:
            raise NormalizationError("Memory estimation inconsistency detected.")
            
        statistics = self._compute_statistics(arr)
        
        processing_time = time.time() - start_time
        
        metadata["NormalizationMethod"] = method_key
        metadata["NormalizationParameters"] = norm_params
        metadata["NormalizationTimestamp"] = datetime.now().isoformat()
        metadata["LayerVersion"] = self.LayerVersion
        metadata["ProcessingTime"] = processing_time
        
        frozen_metadata = MappingProxyType(metadata)
        frozen_statistics = MappingProxyType(statistics)
        frozen_norm_params = MappingProxyType(norm_params)
        
        logger.info(f"Output Shape: {vol_shape}")
        logger.info(f"Output Dtype: {vol_dtype}")
        logger.info(f"Memory Usage: {est_mem_mb:.2f} MB")
        logger.info(f"Normalization Parameters: {norm_params}")
        logger.info(
            f"Statistics: Min={statistics['Minimum']:.4f}, Max={statistics['Maximum']:.4f}, "
            f"Mean={statistics['Mean']:.4f}, DynamicRange={statistics['DynamicRange']:.4f}"
        )
        logger.info(f"Processing Time: {processing_time:.4f} s")
        
        return NormalizedVolume(
            Parent=windowed_volume,
            Volume=arr,
            Metadata=frozen_metadata,
            Statistics=frozen_statistics,
            NormalizationMethod=method_key,
            NormalizationParameters=frozen_norm_params,
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

    def _validate_input_volume(self, windowed_volume: Any) -> None:
        """
        Validates the structure and data types of the input windowed volume.
        """
        if not isinstance(windowed_volume, WindowedVolume):
            raise NormalizationError(f"Input object must be a WindowedVolume, got {type(windowed_volume).__name__}.")
            
        if not hasattr(windowed_volume, "Volume") or not hasattr(windowed_volume, "Metadata"):
            raise NormalizationError("Input object lacks 'Volume' or 'Metadata' attributes.")
            
        if not isinstance(windowed_volume.Volume, np.ndarray):
            raise NormalizationError(f"Volume must be a NumPy array, got {type(windowed_volume.Volume)}")
            
        if windowed_volume.Volume.size == 0:
            raise NormalizationError("Input volume is empty.")
            
        if np.isnan(windowed_volume.Volume).any():
            raise NormalizationError("Input volume contains NaN values.")
            
        if np.isinf(windowed_volume.Volume).any():
            raise NormalizationError("Input volume contains Inf values.")
            
        if not isinstance(windowed_volume.Metadata, Mapping) and not isinstance(windowed_volume.Metadata, dict):
            raise NormalizationError("Metadata must be a dictionary or MappingProxyType.")

    def _compute_statistics(self, volume: np.ndarray) -> dict[str, float]:
        """
        Automatically computes key statistical metrics of the normalized volume.
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
