import copy
import logging
import time
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Any, Tuple, List, Dict, Optional

import numpy as np

from src.pipeline.restoration.expert_execution_manager import ExecutionResult
from src.interfaces.fusion import BaseFusionStrategy, FusionResult, ReliabilityProvider

logger = logging.getLogger(__name__)


class FusionError(Exception):
    """Base exception for fusion engine errors."""
    pass


class InvalidFusionStrategy(FusionError):
    """Raised when an unsupported fusion strategy is requested."""
    pass


class FusionValidationError(FusionError):
    """Raised when fusion inputs fail validation."""
    pass


class DefaultReliabilityProvider(ReliabilityProvider):
    """Reads reliability from expert metadata."""
    def get_reliability(self, expert_name: str, execution_result: ExecutionResult) -> float:
        for res in execution_result.ExpertResults:
            if res.ExpertName == expert_name:
                return float(res.Metadata.get("Reliability", 1.0)) if hasattr(res, "Metadata") else 1.0
        return 1.0


class WeightedAverageFusion(BaseFusionStrategy):
    """
    Fuses expert outputs using their raw routing weights.
    """
    VERSION = "1.0.1"

    def fuse(self, execution_result: Any, ground_truth: Optional[np.ndarray] = None) -> FusionResult:
        start_time = time.time()
        
        self._validate_inputs(execution_result, ground_truth)
        
        expert_results = execution_result.ExpertResults
        metadata = copy.deepcopy(dict(execution_result.Metadata))
        
        raw_weights = {res.ExpertName: float(res.Weight) for res in expert_results}
        normalized_weights = self._normalize_weights(raw_weights)
        
        final_volume = self._perform_fusion(expert_results, normalized_weights)
        
        quality_metrics = self._compute_quality_metrics(final_volume, ground_truth)
        
        processing_time = time.time() - start_time
        strategy_name = "WeightedAverage"
        
        executed_experts = execution_result.ExecutedExperts
        skipped_experts = execution_result.SkippedExperts
        
        metadata["FusionTimestamp"] = datetime.now().isoformat()
        metadata["FusionStrategy"] = strategy_name
        metadata["LayerVersion"] = self.VERSION
        metadata["ProcessingTime"] = processing_time
        
        metadata["ExpertsUsed"] = tuple(normalized_weights.keys())
        metadata["FusionWeights"] = normalized_weights
        metadata["SkippedExperts"] = tuple(skipped_experts)
        metadata["TotalExperts"] = len(executed_experts) + len(skipped_experts)
        metadata["ExecutionStrategy"] = metadata.get("RoutingStrategy", "Unknown")
        
        frozen_metadata = MappingProxyType(metadata)
        frozen_weights = MappingProxyType(normalized_weights)
        frozen_metrics = MappingProxyType(quality_metrics)
        
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"Experts Used: {list(normalized_weights.keys())}")
        logger.info(f"Fusion Weights: {normalized_weights}")
        logger.info(f"Processing Time: {processing_time:.4f} s")
        
        return FusionResult(
            Parent=execution_result,
            FinalVolume=final_volume,
            FusionStrategy=strategy_name,
            ExpertContributions=frozen_weights,
            FusionWeights=frozen_weights,
            QualityMetrics=frozen_metrics,
            Metadata=frozen_metadata,
            ProcessingTime=processing_time,
            LayerVersion=self.VERSION
        )

    def _validate_inputs(self, execution_result: Any, ground_truth: Optional[np.ndarray] = None) -> None:
        if not isinstance(execution_result, ExecutionResult):
            raise FusionValidationError(f"Invalid input type: {type(execution_result).__name__}")
        
        if not execution_result.ExpertResults:
            raise FusionValidationError("At least one ExpertResult is required for fusion.")
            
        base_shape = execution_result.ExpertResults[0].OutputVolume.shape
        base_dtype = execution_result.ExpertResults[0].OutputVolume.dtype
        
        for res in execution_result.ExpertResults:
            if res.OutputVolume.shape != base_shape:
                raise FusionValidationError("Shape inconsistency detected among expert outputs.")
            if res.OutputVolume.dtype != base_dtype:
                raise FusionValidationError("Dtype inconsistency detected among expert outputs.")
                
        if ground_truth is not None:
            if ground_truth.shape != base_shape:
                raise FusionValidationError("Ground truth shape must match expert output shape.")

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            raise FusionValidationError("Sum of fusion weights must be greater than zero.")
            
        normalized = {k: v / total for k, v in weights.items()}
        weight_sum = sum(normalized.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise FusionValidationError(f"Weight sum validation failed: sum={weight_sum}")
        return normalized

    def _perform_fusion(self, expert_results: Tuple[Any, ...], weights: Dict[str, float]) -> np.ndarray:
        base_shape = expert_results[0].OutputVolume.shape
        base_dtype = expert_results[0].OutputVolume.dtype
        
        final_volume = np.zeros(base_shape, dtype=np.float32)
        for res in expert_results:
            final_volume += res.OutputVolume * weights[res.ExpertName]
            
        final_volume = np.clip(final_volume, 0.0, 1.0)
        final_volume = final_volume.astype(base_dtype)
        
        if np.isnan(final_volume).any():
            raise FusionValidationError("Output volume contains NaN values.")
        if np.isinf(final_volume).any():
            raise FusionValidationError("Output volume contains Inf values.")
            
        if final_volume.shape != base_shape:
            raise FusionValidationError("Output volume shape mismatch.")
        if final_volume.dtype != base_dtype:
            raise FusionValidationError("Output volume dtype mismatch.")
            
        final_volume.flags.writeable = False
        return final_volume

    def _compute_quality_metrics(self, volume: np.ndarray, ground_truth: Optional[np.ndarray] = None) -> Dict[str, float]:
        if ground_truth is not None:
            mse = float(np.mean((volume - ground_truth) ** 2))
            mae = float(np.mean(np.abs(volume - ground_truth)))
            psnr = float(10.0 * np.log10(1.0 / (mse + 1e-9)))
            
            mu1 = np.mean(volume)
            mu2 = np.mean(ground_truth)
            var1 = np.var(volume)
            var2 = np.var(ground_truth)
            covar = np.mean((volume - mu1) * (ground_truth - mu2))
            c1 = (0.01) ** 2
            c2 = (0.03) ** 2
            ssim = float(((2 * mu1 * mu2 + c1) * (2 * covar + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)))
            
            return {
                "TrueMSE": mse,
                "TrueMAE": mae,
                "TruePSNR": psnr,
                "TrueSSIM": ssim
            }
        else:
            mean_val = float(np.mean(volume))
            var_val = float(np.var(volume))
            global_noise_level = float(np.clip(var_val * 0.1, 0.0, 1.0))
            
            est_psnr = float(np.clip(50.0 - (global_noise_level * 40.0), 10.0, 100.0))
            est_snr = float(np.clip(40.0 - (global_noise_level * 35.0), 5.0, 80.0))
            est_noise_power = float(global_noise_level * var_val)
            
            return {
                "Mean": mean_val,
                "Variance": var_val,
                "EstimatedPSNR": est_psnr,
                "EstimatedSNR": est_snr,
                "EstimatedNoisePower": est_noise_power
            }


class ConfidenceFusion(WeightedAverageFusion):
    """
    Fuses expert outputs using routing weight * expert confidence.
    """
    def fuse(self, execution_result: Any, ground_truth: Optional[np.ndarray] = None) -> FusionResult:
        start_time = time.time()
        self._validate_inputs(execution_result, ground_truth)
        
        expert_results = execution_result.ExpertResults
        metadata = copy.deepcopy(dict(execution_result.Metadata))
        
        raw_weights = {res.ExpertName: float(res.Weight * res.Confidence) for res in expert_results}
        normalized_weights = self._normalize_weights(raw_weights)
        
        final_volume = self._perform_fusion(expert_results, normalized_weights)
        quality_metrics = self._compute_quality_metrics(final_volume, ground_truth)
        
        processing_time = time.time() - start_time
        strategy_name = "Confidence"
        
        metadata["FusionTimestamp"] = datetime.now().isoformat()
        metadata["FusionStrategy"] = strategy_name
        metadata["LayerVersion"] = self.VERSION
        metadata["ProcessingTime"] = processing_time
        
        executed_experts = execution_result.ExecutedExperts
        skipped_experts = execution_result.SkippedExperts
        metadata["ExpertsUsed"] = tuple(normalized_weights.keys())
        metadata["FusionWeights"] = normalized_weights
        metadata["SkippedExperts"] = tuple(skipped_experts)
        metadata["TotalExperts"] = len(executed_experts) + len(skipped_experts)
        metadata["ExecutionStrategy"] = metadata.get("RoutingStrategy", "Unknown")
        
        frozen_metadata = MappingProxyType(metadata)
        frozen_weights = MappingProxyType(normalized_weights)
        frozen_metrics = MappingProxyType(quality_metrics)
        
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"Experts Used: {list(normalized_weights.keys())}")
        logger.info(f"Fusion Weights: {normalized_weights}")
        logger.info(f"Processing Time: {processing_time:.4f} s")
        
        return FusionResult(
            Parent=execution_result,
            FinalVolume=final_volume,
            FusionStrategy=strategy_name,
            ExpertContributions=frozen_weights,
            FusionWeights=frozen_weights,
            QualityMetrics=frozen_metrics,
            Metadata=frozen_metadata,
            ProcessingTime=processing_time,
            LayerVersion=self.VERSION
        )


class ReliabilityFusion(WeightedAverageFusion):
    """
    Fuses expert outputs using routing weight * expert confidence * expert reliability.
    """
    def __init__(self, reliability_provider: Optional[ReliabilityProvider] = None):
        super().__init__()
        self.provider = reliability_provider or DefaultReliabilityProvider()

    def fuse(self, execution_result: Any, ground_truth: Optional[np.ndarray] = None) -> FusionResult:
        start_time = time.time()
        self._validate_inputs(execution_result, ground_truth)
        
        expert_results = execution_result.ExpertResults
        metadata = copy.deepcopy(dict(execution_result.Metadata))
        
        raw_weights = {}
        for res in expert_results:
            reliability = self.provider.get_reliability(res.ExpertName, execution_result)
            raw_weights[res.ExpertName] = float(res.Weight * res.Confidence * reliability)
            
        normalized_weights = self._normalize_weights(raw_weights)
        
        final_volume = self._perform_fusion(expert_results, normalized_weights)
        quality_metrics = self._compute_quality_metrics(final_volume, ground_truth)
        
        processing_time = time.time() - start_time
        strategy_name = "Reliability"
        
        metadata["FusionTimestamp"] = datetime.now().isoformat()
        metadata["FusionStrategy"] = strategy_name
        metadata["LayerVersion"] = self.VERSION
        metadata["ProcessingTime"] = processing_time
        
        executed_experts = execution_result.ExecutedExperts
        skipped_experts = execution_result.SkippedExperts
        metadata["ExpertsUsed"] = tuple(normalized_weights.keys())
        metadata["FusionWeights"] = normalized_weights
        metadata["SkippedExperts"] = tuple(skipped_experts)
        metadata["TotalExperts"] = len(executed_experts) + len(skipped_experts)
        metadata["ExecutionStrategy"] = metadata.get("RoutingStrategy", "Unknown")
        
        frozen_metadata = MappingProxyType(metadata)
        frozen_weights = MappingProxyType(normalized_weights)
        frozen_metrics = MappingProxyType(quality_metrics)
        
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"Experts Used: {list(normalized_weights.keys())}")
        logger.info(f"Fusion Weights: {normalized_weights}")
        logger.info(f"Processing Time: {processing_time:.4f} s")
        
        return FusionResult(
            Parent=execution_result,
            FinalVolume=final_volume,
            FusionStrategy=strategy_name,
            ExpertContributions=frozen_weights,
            FusionWeights=frozen_weights,
            QualityMetrics=frozen_metrics,
            Metadata=frozen_metadata,
            ProcessingTime=processing_time,
            LayerVersion=self.VERSION
        )


class RegionAdaptiveFusion(BaseFusionStrategy):
    """
    Interface for Region-Adaptive Fusion strategy.
    Not implemented.
    """
    def fuse(self, execution_result: Any, ground_truth: Optional[np.ndarray] = None) -> FusionResult:
        raise NotImplementedError("RegionAdaptiveFusion is not yet implemented.")


class LearnedFusion(BaseFusionStrategy):
    """
    Interface for Learned Fusion strategy (e.g. CNN based).
    Not implemented.
    """
    def fuse(self, execution_result: Any, ground_truth: Optional[np.ndarray] = None) -> FusionResult:
        raise NotImplementedError("LearnedFusion is not yet implemented.")


class FusionEngine:
    """
    Engine to manage and dispatch the fusion process across available strategies.
    """
    def __init__(self, reliability_provider: Optional[ReliabilityProvider] = None):
        self.strategies = {
            "weightedaverage": WeightedAverageFusion(),
            "confidence": ConfidenceFusion(),
            "reliability": ReliabilityFusion(reliability_provider),
            "regionadaptive": RegionAdaptiveFusion(),
            "learned": LearnedFusion()
        }

    def fuse(
        self, 
        execution_result: Any, 
        strategy: str = "confidence",
        ground_truth: Optional[np.ndarray] = None
    ) -> FusionResult:
        """
        Fuses expert outputs using the specified strategy.
        
        Args:
            execution_result: The ExecutionResult from Layer 9.
            strategy: The fusion strategy to use.
            ground_truth: Optional ground truth for training mode evaluation.
            
        Returns:
            FusionResult: Immutable dataclass containing the final fused volume.
            
        Raises:
            InvalidFusionStrategy: If the requested strategy is not supported.
        """
        strategy_key = strategy.lower()
        if strategy_key not in self.strategies:
            raise InvalidFusionStrategy(f"Fusion strategy {strategy} is not supported.")
            
        strategy_instance = self.strategies[strategy_key]
        return strategy_instance.fuse(execution_result, ground_truth)
