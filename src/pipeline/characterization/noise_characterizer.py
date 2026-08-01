import copy
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Any, Tuple, List, Dict, Optional

import numpy as np

from src.pipeline.noise.noise_generator import NoisyVolume

logger = logging.getLogger(__name__)


class CharacterizationError(Exception):
    """Base exception for noise characterization errors."""
    pass


@dataclass(frozen=True)
class CharacterizationReport:
    """
    Immutable representation of noise characterization results.
    """
    Parent: NoisyVolume
    NoiseProbabilities: Mapping[str, float]
    NoiseSeverities: Mapping[str, str]
    NoiseRanking: Tuple[str, ...]
    DominantNoise: str
    DominantConfidence: float
    ActivatedCandidates: Tuple[str, ...]
    GlobalNoiseLevel: float
    ImageQualityScore: float
    EstimatedPSNR: float
    EstimatedSNR: float
    EstimatedNoisePower: float
    ExtractedFeatures: Mapping[str, float]
    Statistics: Mapping[str, float]
    Metadata: Mapping[str, Any]
    ProcessingTime: float
    LayerVersion: str


class CTNoiseCharacterizer:
    """
    Engine for characterizing noise in CT volumes.
    This is the exclusive layer responsible for noise estimation and classification.
    """
    
    VERSION: str = "1.0.1"
    
    SUPPORTED_NOISE_TYPES = {
        "gaussian", "poisson", "speckle", 
        "saltpepper", "motionblur", "ringartifact"
    }

    def characterize(self, noisy_volume: Any, confidence_buffer: float = 0.10) -> CharacterizationReport:
        """
        Characterizes noise in a NoisyVolume.
        
        Args:
            noisy_volume: The NoisyVolume object from Layer 6.
            confidence_buffer: Range to select activated candidates.
            
        Returns:
            CharacterizationReport: Immutable dataclass containing characterization results.
            
        Raises:
            CharacterizationError: For general characterization failures.
        """
        start_time = time.time()
        
        self._validate_input_volume(noisy_volume)
        
        arr = noisy_volume.Volume.astype(np.float32)
        metadata = copy.deepcopy(dict(noisy_volume.Metadata))
        
        patient_id = metadata.get("PatientID", "UNKNOWN")
        series_uid = metadata.get("SeriesInstanceUID", "UNKNOWN")
        
        logger.info(f"PatientID: {patient_id}")
        logger.info(f"SeriesUID: {series_uid}")
        
        features = self._extract_features(arr)
        confidences = self._compute_confidences(features)
        
        ranked_noises = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
        noise_ranking = tuple(k for k, v in ranked_noises)
        
        max_conf = ranked_noises[0][1]
        dominant_noise = ranked_noises[0][0]
        
        candidates = tuple(
            k for k, v in confidences.items() if v >= max_conf - confidence_buffer
        )
        
        severities = {k: self._map_severity(v) for k, v in confidences.items()}
        
        global_noise_level = float(np.mean(list(confidences.values())))
        
        est_psnr = float(np.clip(50.0 - (global_noise_level * 40.0), 10.0, 100.0))
        est_snr = float(np.clip(40.0 - (global_noise_level * 35.0), 5.0, 80.0))
        est_noise_power = float(global_noise_level * features["Variance"])
        
        if "PSNR" in metadata:
            est_psnr = float(metadata["PSNR"])
        if "SNR" in metadata:
            est_snr = float(metadata["SNR"])
        if "EstimatedNoisePower" in metadata:
            est_noise_power = float(metadata["EstimatedNoisePower"])
            
        v_norm = np.clip(1.0 - (features["Variance"] * 10.0), 0.0, 1.0)
        e_norm = np.clip(1.0 - (features["Entropy"] / 8.0), 0.0, 1.0)
        hf_norm = np.clip(1.0 - features["HighFrequencyRatio"], 0.0, 1.0)
        psnr_norm = np.clip((est_psnr - 10.0) / 90.0, 0.0, 1.0)
        snr_norm = np.clip((est_snr - 5.0) / 75.0, 0.0, 1.0)
        
        iq_score = float(0.1 * v_norm + 0.1 * e_norm + 0.1 * hf_norm + 0.4 * psnr_norm + 0.3 * snr_norm)
            
        if dominant_noise not in confidences:
            raise CharacterizationError("DominantNoise does not exist in NoiseProbabilities.")
            
        for cand in candidates:
            if cand not in self.SUPPORTED_NOISE_TYPES:
                raise CharacterizationError(f"Activated candidate {cand} is not a supported noise type.")
                
        basic_stats = self._compute_statistics(arr)
        
        processing_time = time.time() - start_time
        
        metadata["CharacterizationTimestamp"] = datetime.now().isoformat()
        metadata["LayerVersion"] = self.VERSION
        metadata["ProcessingTime"] = processing_time
        
        frozen_metadata = MappingProxyType(metadata)
        frozen_statistics = MappingProxyType(basic_stats)
        frozen_probabilities = MappingProxyType(confidences)
        frozen_severities = MappingProxyType(severities)
        frozen_features = MappingProxyType(features)
        
        logger.info(f"Dominant Noise: {dominant_noise}")
        logger.info(f"Dominant Confidence: {max_conf:.4f}")
        logger.info(f"Activated Candidates: {candidates}")
        logger.info(f"Noise Ranking: {noise_ranking}")
        logger.info(f"Confidence Buffer: {confidence_buffer}")
        logger.info(f"Global Noise Level: {global_noise_level:.4f}")
        logger.info(f"Extracted Features Summary: Var={features['Variance']:.4f}, Ent={features['Entropy']:.4f}, PSNR={est_psnr:.2f}")
        logger.info(f"Processing Time: {processing_time:.4f} s")
        
        return CharacterizationReport(
            Parent=noisy_volume,
            NoiseProbabilities=frozen_probabilities,
            NoiseSeverities=frozen_severities,
            NoiseRanking=noise_ranking,
            DominantNoise=dominant_noise,
            DominantConfidence=max_conf,
            ActivatedCandidates=candidates,
            GlobalNoiseLevel=global_noise_level,
            ImageQualityScore=iq_score,
            EstimatedPSNR=est_psnr,
            EstimatedSNR=est_snr,
            EstimatedNoisePower=est_noise_power,
            ExtractedFeatures=frozen_features,
            Statistics=frozen_statistics,
            Metadata=frozen_metadata,
            ProcessingTime=processing_time,
            LayerVersion=self.VERSION
        )

    def _validate_input_volume(self, noisy_volume: Any) -> None:
        if not isinstance(noisy_volume, NoisyVolume):
            raise CharacterizationError(
                f"Input object must be a NoisyVolume, got {type(noisy_volume).__name__}."
            )
            
        if not hasattr(noisy_volume, "Volume") or not hasattr(noisy_volume, "Metadata"):
            raise CharacterizationError("Input object lacks 'Volume' or 'Metadata' attributes.")
            
        if not isinstance(noisy_volume.Volume, np.ndarray):
            raise CharacterizationError(f"Volume must be a NumPy array, got {type(noisy_volume.Volume)}")
            
        if noisy_volume.Volume.size == 0:
            raise CharacterizationError("Input volume is empty.")
            
        if np.isnan(noisy_volume.Volume).any():
            raise CharacterizationError("Input volume contains NaN values.")
            
        if np.isinf(noisy_volume.Volume).any():
            raise CharacterizationError("Input volume contains Inf values.")
            
        if not isinstance(noisy_volume.Metadata, Mapping) and not isinstance(noisy_volume.Metadata, dict):
            raise CharacterizationError("Metadata must be a dictionary or MappingProxyType.")

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

    def _extract_features(self, volume: np.ndarray) -> dict[str, float]:
        mean = float(np.mean(volume))
        var = float(np.var(volume))
        std = float(np.std(volume))
        
        hist, _ = np.histogram(volume, bins=256, range=(0.0, 1.0), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist + 1e-9)))
        
        diff = volume - mean
        if std > 0:
            skewness = float(np.mean(diff**3) / (std**3))
            kurtosis = float(np.mean(diff**4) / (std**4)) - 3.0
        else:
            skewness = 0.0
            kurtosis = 0.0
            
        if volume.ndim >= 3:
            central_slice = volume[volume.shape[0] // 2]
        else:
            central_slice = volume
            
        fft = np.fft.fft2(central_slice)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        energy = np.sum(magnitude**2)
        
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[0:h, 0:w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_max = min(cy, cx)
        
        low_freq_mask = r < (0.25 * r_max)
        high_freq_mask = r >= (0.25 * r_max)
        
        low_freq_energy = np.sum(magnitude[low_freq_mask]**2)
        high_freq_energy = np.sum(magnitude[high_freq_mask]**2)
        
        hf_ratio = float(high_freq_energy / (energy + 1e-9))
        lf_ratio = float(low_freq_energy / (energy + 1e-9))
        
        diff_y = np.abs(np.diff(central_slice, axis=0))
        diff_x = np.abs(np.diff(central_slice, axis=1))
        edge_density = float((np.mean(diff_y) + np.mean(diff_x)) / 2.0)
        
        local_var = float(edge_density ** 2)
        
        p99 = float(np.percentile(volume, 99))
        p01 = float(np.percentile(volume, 1))
        impulse_score = float(np.mean((volume > p99) | (volume < p01)))
        
        diff_y_var = np.var(diff_y)
        diff_x_var = np.var(diff_x)
        motion_score = float(abs(diff_y_var - diff_x_var) / (diff_y_var + diff_x_var + 1e-9))
        
        ring_score = float(np.var(np.mean(magnitude, axis=0)))
        
        return {
            "Mean": mean,
            "Variance": var,
            "Entropy": entropy,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
            "FFTEnergy": float(energy),
            "HighFrequencyRatio": hf_ratio,
            "LowFrequencyRatio": lf_ratio,
            "EdgeDensity": edge_density,
            "LocalVariance": local_var,
            "RingScore": ring_score,
            "MotionScore": motion_score,
            "ImpulseScore": impulse_score
        }

    def _compute_confidences(self, features: dict[str, float]) -> dict[str, float]:
        raw_scores = {
            "gaussian": self._compute_gaussian_confidence(features),
            "poisson": self._compute_poisson_confidence(features),
            "speckle": self._compute_speckle_confidence(features),
            "saltpepper": self._compute_saltpepper_confidence(features),
            "motionblur": self._compute_motionblur_confidence(features),
            "ringartifact": self._compute_ringartifact_confidence(features)
        }
        
        # Normalize jointly
        total_score = sum(raw_scores.values())
        if total_score > 0:
            return {k: v / total_score for k, v in raw_scores.items()}
        else:
            n = len(raw_scores)
            return {k: 1.0 / n for k in raw_scores.keys()}
            
    def _compute_gaussian_confidence(self, features: dict[str, float]) -> float:
        """
        Computes raw confidence for Gaussian noise.
        - Variance contributes positively as Gaussian noise directly increases global variance.
        - HighFrequencyRatio contributes positively because white noise affects all frequencies equally.
        """
        return max(0.0, features["Variance"] * 5.0 + features["HighFrequencyRatio"] * 0.5)

    def _compute_poisson_confidence(self, features: dict[str, float]) -> float:
        """
        Computes raw confidence for Poisson noise.
        - Variance contributes positively, but generally lower than purely additive Gaussian noise.
        - Entropy contributes positively due to the signal-dependent nature adding widespread randomness.
        """
        return max(0.0, features["Variance"] * 2.5 + features["Entropy"] * 0.05)

    def _compute_speckle_confidence(self, features: dict[str, float]) -> float:
        """
        Computes raw confidence for Speckle noise.
        - EdgeDensity contributes positively as speckle heavily distorts local gradients.
        - Variance contributes positively as multiplicative noise scales with signal intensity.
        """
        return max(0.0, features["EdgeDensity"] * 1.5 + features["Variance"] * 4.0)

    def _compute_saltpepper_confidence(self, features: dict[str, float]) -> float:
        """
        Computes raw confidence for Salt & Pepper noise.
        - ImpulseScore contributes positively as this noise creates extreme outliers.
        - Kurtosis contributes positively due to heavy tails from impulses.
        """
        return max(0.0, features["ImpulseScore"] * 15.0 + (features["Kurtosis"] / 20.0))

    def _compute_motionblur_confidence(self, features: dict[str, float]) -> float:
        """
        Computes raw confidence for Motion Blur.
        - MotionScore contributes positively due to directional variance differences.
        - HighFrequencyRatio contributes negatively since blur acts as a low-pass filter.
        """
        return max(0.0, features["MotionScore"] * 1.5 + max(0.0, 1.0 - features["HighFrequencyRatio"]) * 0.4)

    def _compute_ringartifact_confidence(self, features: dict[str, float]) -> float:
        """
        Computes raw confidence for Ring Artifacts.
        - RingScore contributes positively by detecting structured concentric patterns in the FFT.
        - EdgeDensity contributes positively due to artificial structural edges.
        """
        return max(0.0, features["RingScore"] * 0.001 + features["EdgeDensity"] * 0.3)

    def _map_severity(self, confidence: float) -> str:
        if confidence < 0.2:
            return "VeryLow"
        elif confidence < 0.4:
            return "Low"
        elif confidence < 0.6:
            return "Medium"
        elif confidence < 0.8:
            return "High"
        else:
            return "Extreme"
