import copy
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Any, Tuple, List, Dict, Optional

import numpy as np

from src.pipeline.preprocessing.normalization.normalizer import NormalizedVolume

logger = logging.getLogger(__name__)


class NoiseGenerationError(Exception):
    """Base exception for noise generation errors."""
    pass


class InvalidNoiseTypeError(NoiseGenerationError):
    """Raised when an unsupported noise type is requested."""
    pass


class InvalidIntensityError(NoiseGenerationError):
    """Raised when an unsupported intensity level is requested."""
    pass


@dataclass(frozen=True)
class NoisyVolume:
    """
    Immutable representation of a synthetic noise injected CT volume.
    """
    Parent: NormalizedVolume
    Volume: np.ndarray
    Metadata: Mapping[str, Any]
    Statistics: Mapping[str, float]
    NoiseHistory: Tuple[Mapping[str, Any], ...]
    NoiseTypes: Tuple[str, ...]
    NoiseParameters: Mapping[str, Any]
    NoiseIntensity: str
    RandomSeed: int
    PSNR: float
    SNR: float
    EstimatedNoisePower: float
    VolumeShape: Tuple[int, ...]
    VolumeDtype: str
    EstimatedMemoryMB: float
    ProcessingTime: float
    LayerVersion: str


class CTNoiseGenerator:
    """
    Engine for generating and applying synthetic noise to CT volumes.
    This is the exclusive layer responsible for noise injection.
    """
    
    VERSION: str = "1.0.1"
    
    SUPPORTED_NOISE_TYPES = {
        "gaussian", "poisson", "speckle", 
        "saltpepper", "motionblur", "ringartifact"
    }
    
    SUPPORTED_INTENSITIES = {
        "very_low", "low", "medium", "high", "extreme"
    }
    
    INTENSITY_PARAMS = {
        "very_low": {
            "gaussian": {"std": 0.01},
            "poisson": {"lam": 100.0},
            "speckle": {"var": 0.005},
            "saltpepper": {"prob": 0.01},
            "motionblur": {"kernel_length": 5, "angle_range": (-10.0, 10.0)},
            "ringartifact": {"num_rings": 2, "max_amplitude": 0.02}
        },
        "low": {
            "gaussian": {"std": 0.03},
            "poisson": {"lam": 50.0},
            "speckle": {"var": 0.01},
            "saltpepper": {"prob": 0.03},
            "motionblur": {"kernel_length": 11, "angle_range": (-20.0, 20.0)},
            "ringartifact": {"num_rings": 5, "max_amplitude": 0.05}
        },
        "medium": {
            "gaussian": {"std": 0.05},
            "poisson": {"lam": 20.0},
            "speckle": {"var": 0.03},
            "saltpepper": {"prob": 0.05},
            "motionblur": {"kernel_length": 21, "angle_range": (-45.0, 45.0)},
            "ringartifact": {"num_rings": 10, "max_amplitude": 0.10}
        },
        "high": {
            "gaussian": {"std": 0.10},
            "poisson": {"lam": 10.0},
            "speckle": {"var": 0.05},
            "saltpepper": {"prob": 0.10},
            "motionblur": {"kernel_length": 31, "angle_range": (-90.0, 90.0)},
            "ringartifact": {"num_rings": 20, "max_amplitude": 0.15}
        },
        "extreme": {
            "gaussian": {"std": 0.20},
            "poisson": {"lam": 5.0},
            "speckle": {"var": 0.10},
            "saltpepper": {"prob": 0.20},
            "motionblur": {"kernel_length": 41, "angle_range": (-180.0, 180.0)},
            "ringartifact": {"num_rings": 30, "max_amplitude": 0.25}
        }
    }

    def generate(
        self,
        normalized_volume: Any,
        noise_types: Optional[List[str]] = None,
        intensity: str = "medium",
        random_seed: int = 42
    ) -> NoisyVolume:
        """
        Injects synthetic noise into a normalized CT volume.
        
        Args:
            normalized_volume: The NormalizedVolume object from Layer 4.
            noise_types: List of noise types to apply sequentially.
            intensity: Overall intensity level (very_low, low, medium, high, extreme).
            random_seed: Deterministic random seed.
            
        Returns:
            NoisyVolume: An immutable dataclass containing the noisy volume.
            
        Raises:
            NoiseGenerationError: For general noise generation failures.
            InvalidNoiseTypeError: If an unknown noise type is requested.
            InvalidIntensityError: If the intensity level is unsupported.
        """
        start_time = time.time()
        
        self._validate_input_volume(normalized_volume)
        
        if noise_types is None:
            noise_types = ["gaussian"]
            
        intensity_key = intensity.lower()
        if intensity_key not in self.SUPPORTED_INTENSITIES:
            raise InvalidIntensityError(
                f"Unsupported intensity: {intensity}. Supported: {list(self.SUPPORTED_INTENSITIES)}"
            )
            
        rng = np.random.default_rng(seed=random_seed)
        
        metadata: dict[str, Any] = copy.deepcopy(dict(normalized_volume.Metadata))
        orig_volume: np.ndarray = normalized_volume.Volume
        
        patient_id = metadata.get("PatientID", "UNKNOWN")
        series_uid = metadata.get("SeriesInstanceUID", "UNKNOWN")
        
        logger.info(f"PatientID: {patient_id}")
        logger.info(f"SeriesUID: {series_uid}")
        logger.info(f"Input Shape: {orig_volume.shape}")
        logger.info(f"Noise Types: {noise_types}")
        logger.info(f"Intensity: {intensity_key}")
        logger.info(f"Random Seed: {random_seed}")
        
        arr = orig_volume.astype(np.float32)
        history = []
        applied_types = []
        
        for n_type in noise_types:
            n_key = n_type.lower()
            if n_key not in self.SUPPORTED_NOISE_TYPES:
                raise InvalidNoiseTypeError(
                    f"Unsupported noise type: {n_type}. Supported: {self.SUPPORTED_NOISE_TYPES}"
                )
                
            n_params = self.INTENSITY_PARAMS[intensity_key][n_key]
            
            if n_key == "gaussian":
                arr, used_params = self._apply_gaussian(arr, n_params, rng)
            elif n_key == "poisson":
                arr, used_params = self._apply_poisson(arr, n_params, rng)
            elif n_key == "speckle":
                arr, used_params = self._apply_speckle(arr, n_params, rng)
            elif n_key == "saltpepper":
                arr, used_params = self._apply_saltpepper(arr, n_params, rng)
            elif n_key == "motionblur":
                arr, used_params = self._apply_motionblur(arr, n_params, rng)
            elif n_key == "ringartifact":
                arr, used_params = self._apply_ringartifact(arr, n_params, rng)
                
            op_record = {
                "NoiseType": n_key,
                "Intensity": intensity_key,
                "Parameters": used_params,
                "Timestamp": datetime.now().isoformat()
            }
                
            history.append(MappingProxyType(op_record))
            applied_types.append(n_key)
            
        arr = np.clip(arr, 0.0, 1.0)
        arr.flags.writeable = False
        
        vol_shape = arr.shape
        vol_dtype = str(arr.dtype)
        est_mem_mb = float(arr.nbytes / (1024 * 1024))
        
        if vol_shape != orig_volume.shape:
            raise NoiseGenerationError(f"Shape inconsistency detected: {vol_shape} != {orig_volume.shape}")
        if vol_dtype != "float32":
            raise NoiseGenerationError(f"Dtype inconsistency detected: {vol_dtype} != float32")
        expected_bytes = np.prod(vol_shape) * 4
        if abs(est_mem_mb - (expected_bytes / (1024 * 1024))) > 1e-6:
            raise NoiseGenerationError("Memory estimation inconsistency detected.")
        if np.isnan(arr).any():
            raise NoiseGenerationError("Resulting volume contains NaN values.")
        if np.isinf(arr).any():
            raise NoiseGenerationError("Resulting volume contains Inf values.")
        if arr.flags.writeable:
            raise NoiseGenerationError("Resulting volume array is not read-only.")
        
        psnr, snr, noise_power = self._compute_psnr_snr_power(orig_volume, arr)
        statistics = self._compute_statistics(arr)
        
        processing_time = time.time() - start_time
        
        noise_params = {
            "Intensity": intensity_key,
            "RandomSeed": random_seed
        }
        
        metadata["NoiseTypes"] = tuple(applied_types)
        metadata["NoiseParameters"] = noise_params
        metadata["NoiseHistory"] = tuple(history)
        metadata["RandomSeed"] = random_seed
        metadata["PSNR"] = psnr
        metadata["SNR"] = snr
        metadata["EstimatedNoisePower"] = noise_power
        metadata["NoiseGenerationTimestamp"] = datetime.now().isoformat()
        metadata["LayerVersion"] = self.VERSION
        metadata["ProcessingTime"] = processing_time
        
        frozen_metadata = MappingProxyType(metadata)
        frozen_statistics = MappingProxyType(statistics)
        frozen_noise_params = MappingProxyType(noise_params)
        frozen_history = tuple(history)
        frozen_types = tuple(applied_types)
        
        logger.info(f"PSNR: {psnr:.2f} dB, SNR: {snr:.2f} dB, Noise Power: {noise_power:.6f}")
        logger.info(f"Noise History: {[h['NoiseType'] for h in history]}")
        logger.info(f"Estimated Memory: {est_mem_mb:.2f} MB")
        logger.info(f"Dynamic Range: {statistics['DynamicRange']:.4f}")
        processing_speed = est_mem_mb / processing_time if processing_time > 0 else 0
        logger.info(f"Processing Speed: {processing_speed:.2f} MB/s")
        logger.info(f"Processing Time: {processing_time:.4f} s")
        
        return NoisyVolume(
            Parent=normalized_volume,
            Volume=arr,
            Metadata=frozen_metadata,
            Statistics=frozen_statistics,
            NoiseHistory=frozen_history,
            NoiseTypes=frozen_types,
            NoiseParameters=frozen_noise_params,
            NoiseIntensity=intensity_key,
            RandomSeed=random_seed,
            PSNR=psnr,
            SNR=snr,
            EstimatedNoisePower=noise_power,
            VolumeShape=vol_shape,
            VolumeDtype=vol_dtype,
            EstimatedMemoryMB=est_mem_mb,
            ProcessingTime=processing_time,
            LayerVersion=self.VERSION
        )

    def _validate_input_volume(self, normalized_volume: Any) -> None:
        if not isinstance(normalized_volume, NormalizedVolume):
            raise NoiseGenerationError(
                f"Input object must be a NormalizedVolume, got {type(normalized_volume).__name__}."
            )
            
        if not hasattr(normalized_volume, "Volume") or not hasattr(normalized_volume, "Metadata"):
            raise NoiseGenerationError("Input object lacks 'Volume' or 'Metadata' attributes.")
            
        if not isinstance(normalized_volume.Volume, np.ndarray):
            raise NoiseGenerationError(f"Volume must be a NumPy array, got {type(normalized_volume.Volume)}")
            
        if normalized_volume.Volume.size == 0:
            raise NoiseGenerationError("Input volume is empty.")
            
        if np.isnan(normalized_volume.Volume).any():
            raise NoiseGenerationError("Input volume contains NaN values.")
            
        if np.isinf(normalized_volume.Volume).any():
            raise NoiseGenerationError("Input volume contains Inf values.")
            
        if not isinstance(normalized_volume.Metadata, Mapping) and not isinstance(normalized_volume.Metadata, dict):
            raise NoiseGenerationError("Metadata must be a dictionary or MappingProxyType.")

    def _compute_psnr_snr_power(self, orig: np.ndarray, noisy: np.ndarray) -> Tuple[float, float, float]:
        mse = float(np.mean((noisy - orig) ** 2))
        noise_power = mse
        
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = float(10.0 * np.log10(1.0 / mse))
            
        var_signal = float(np.var(orig))
        var_noise = float(np.var(noisy - orig))
        
        if var_noise == 0:
            snr = float('inf')
        elif var_signal == 0:
            snr = 0.0
        else:
            snr = float(10.0 * np.log10(var_signal / var_noise))
            
        return psnr, snr, noise_power

    def _compute_statistics(self, volume: np.ndarray) -> dict[str, float]:
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

    # Noise Applicators

    def _apply_gaussian(self, arr: np.ndarray, params: dict, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
        std = params["std"]
        noise = rng.normal(0.0, std, size=arr.shape).astype(np.float32)
        out = arr + noise
        return out, {"StandardDeviation": float(std)}

    def _apply_poisson(self, arr: np.ndarray, params: dict, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
        lam = params["lam"]
        arr_clipped = np.clip(arr * lam, 0, None)
        out = (rng.poisson(arr_clipped) / lam).astype(np.float32)
        return out, {"Lambda": float(lam)}

    def _apply_speckle(self, arr: np.ndarray, params: dict, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
        var = params["var"]
        std = np.sqrt(var)
        noise = rng.normal(0.0, std, size=arr.shape).astype(np.float32)
        out = arr + arr * noise
        return out, {"Variance": float(var), "StandardDeviation": float(std)}

    def _apply_saltpepper(self, arr: np.ndarray, params: dict, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
        prob = params["prob"]
        out = arr.copy()
        r = rng.random(out.shape).astype(np.float32)
        salt_prob = prob / 2.0
        pepper_prob = prob / 2.0
        out[r < pepper_prob] = 0.0
        out[(r >= pepper_prob) & (r < prob)] = 1.0
        return out, {"SaltProbability": float(salt_prob), "PepperProbability": float(pepper_prob)}

    def _apply_motionblur(self, arr: np.ndarray, params: dict, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
        k_len = params["kernel_length"]
        angle_min, angle_max = params["angle_range"]
        angle = float(rng.uniform(angle_min, angle_max))
        rad = np.deg2rad(angle)
        
        out = np.zeros_like(arr)
        for i in range(k_len):
            offset = i - k_len // 2
            dy = int(np.round(offset * np.sin(rad)))
            dx = int(np.round(offset * np.cos(rad)))
            out += np.roll(arr, shift=(dy, dx), axis=(-2, -1))
        out /= k_len
        
        return out, {"KernelLength": int(k_len), "Angle": float(angle)}

    def _apply_ringartifact(self, arr: np.ndarray, params: dict, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
        num_rings = params["num_rings"]
        max_amp = params["max_amplitude"]
        
        if arr.ndim < 2:
            return arr, {}
            
        shape = arr.shape
        y_dim, x_dim = shape[-2], shape[-1]
        y, x = np.ogrid[0:y_dim, 0:x_dim]
        cy, cx = y_dim / 2.0, x_dim / 2.0
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        rings_mask = np.zeros_like(r, dtype=np.float32)
        max_radius = min(cx, cy)
        
        radii = rng.uniform(0, max_radius, size=num_rings).tolist()
        widths = rng.uniform(0.5, 3.0, size=num_rings).tolist()
        amplitudes = rng.uniform(-max_amp, max_amp, size=num_rings).tolist()
        
        for rad, w, amp in zip(radii, widths, amplitudes):
            ring = amp * np.exp(-((r - rad)**2) / (2 * w**2))
            rings_mask += ring
            
        broadcast_shape = [1] * (arr.ndim - 2) + [y_dim, x_dim]
        rings_mask = rings_mask.reshape(broadcast_shape).astype(np.float32)
        
        out = arr + rings_mask
        
        used_params = {
            "NumberOfRings": int(num_rings),
            "MaxAmplitude": float(max_amp),
            "RingRadii": radii,
            "RingWidths": widths,
            "RingAmplitudes": amplitudes
        }
        return out, used_params
