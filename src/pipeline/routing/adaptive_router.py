import copy
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Any, Tuple, List, Dict, Optional

from src.pipeline.characterization.noise_characterizer import CharacterizationReport

logger = logging.getLogger(__name__)


class RoutingError(Exception):
    """Base exception for routing engine errors."""
    pass


@dataclass(frozen=True)
class ExpertExecution:
    """
    Immutable representation of an individual expert's execution plan.
    """
    ExpertName: str
    Priority: int
    Weight: float
    Confidence: float
    Execute: bool
    Reason: str


@dataclass(frozen=True)
class RoutingDecision:
    """
    Immutable representation of an expert routing plan.
    """
    Parent: CharacterizationReport
    ExpertPlan: Tuple[ExpertExecution, ...]
    PrimaryExpert: str
    SecondaryExperts: Tuple[str, ...]
    ExecutionOrder: Tuple[str, ...]
    ExpertWeights: Mapping[str, float]
    RoutingStrategy: str
    RoutingConfidence: float
    RequiresFusion: bool
    FallbackMode: bool
    Metadata: Mapping[str, Any]
    ProcessingTime: float
    LayerVersion: str


class AdaptiveRouter:
    """
    Engine for converting characterization reports into executable expert routing plans.
    This layer decides expert routing without performing actual image denoising.
    """
    
    VERSION: str = "1.0.1"
    
    EXPERT_MAPPING = {
        "gaussian": "GaussianExpert",
        "poisson": "PoissonExpert",
        "speckle": "SpeckleExpert",
        "saltpepper": "SaltPepperExpert",
        "motionblur": "MotionBlurExpert",
        "ringartifact": "RingArtifactExpert"
    }

    def route(
        self, 
        characterization_report: Any, 
        fallback_threshold: float = 0.05,
        default_expert: str = "gaussian",
        confidence_buffer: float = 0.10
    ) -> RoutingDecision:
        """
        Generates an expert routing plan based on the characterization report.
        
        Args:
            characterization_report: The CharacterizationReport from Layer 7.
            fallback_threshold: Minimum confidence required to avoid fallback mode.
            default_expert: The default expert to use in fallback mode.
            confidence_buffer: The confidence buffer value to log/store in metadata.
            
        Returns:
            RoutingDecision: Immutable dataclass containing the routing plan.
            
        Raises:
            RoutingError: If validation fails or routing logic encounters an inconsistency.
        """
        start_time = time.time()
        
        self._validate_input(characterization_report)
        
        metadata = copy.deepcopy(dict(characterization_report.Metadata))
        probs = dict(characterization_report.NoiseProbabilities)
        
        if not probs:
            raise RoutingError("Empty probability map provided.")
            
        for k, v in probs.items():
            if not isinstance(v, (int, float)) or v < 0.0 or v > 1.0:
                raise RoutingError(f"Invalid confidence value for {k}: {v}")
                
        max_conf = max(probs.values())
        
        if max_conf < fallback_threshold:
            strategy = "Fallback"
            fallback_mode = True
            requires_fusion = False
            default_key = default_expert.lower()
            if default_key not in self.EXPERT_MAPPING:
                raise RoutingError(f"Default expert {default_key} is not supported.")
            selected_noises = [default_key]
            weights = {self.EXPERT_MAPPING[default_key]: 1.0}
        else:
            fallback_mode = False
            candidates = characterization_report.ActivatedCandidates
            if not candidates:
                raise RoutingError("No activated candidates found despite sufficient confidence.")
                
            if len(candidates) == 1:
                strategy = "SingleExpert"
                requires_fusion = False
                selected_noises = [candidates[0]]
                weights = {self.EXPERT_MAPPING[candidates[0]]: 1.0}
            else:
                selected_noises = sorted(candidates, key=lambda c: probs[c], reverse=True)
                total_conf = sum(probs[n] for n in selected_noises)
                if total_conf > 0:
                    weights = {self.EXPERT_MAPPING[n]: float(probs[n] / total_conf) for n in selected_noises}
                else:
                    eq_weight = 1.0 / len(selected_noises)
                    weights = {self.EXPERT_MAPPING[n]: eq_weight for n in selected_noises}
                    
                weights_vals = list(weights.values())
                # If weights are nearly identical (difference < 0.15)
                if max(weights_vals) - min(weights_vals) < 0.15:
                    strategy = "WeightedFusion"
                    requires_fusion = True
                else:
                    strategy = "MultiExpert"
                    requires_fusion = False
                
        selected_experts = [self.EXPERT_MAPPING[n] for n in selected_noises]
        primary_expert = selected_experts[0]
        secondary_experts = tuple(selected_experts[1:])
        execution_order = tuple(selected_experts)
        
        if len(set(execution_order)) != len(execution_order):
            raise RoutingError("Duplicate experts detected in execution order.")
            
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise RoutingError(f"Expert weights do not sum to 1.0 (Sum: {weight_sum})")
            
        expert_plan_list = []
        routing_confidence = 0.0
        
        for idx, expert_name in enumerate(execution_order):
            noise_type = selected_noises[idx]
            conf = 0.0 if fallback_mode else float(probs[noise_type])
            weight = weights[expert_name]
            reason = f"Fallback default expert." if fallback_mode else f"Activated by confidence {conf:.4f}"
            
            routing_confidence += conf * weight
            
            plan_entry = ExpertExecution(
                ExpertName=expert_name,
                Priority=idx + 1,
                Weight=weight,
                Confidence=conf,
                Execute=True,
                Reason=reason
            )
            expert_plan_list.append(plan_entry)
            
        expert_plan = tuple(expert_plan_list)
        
        # Validations
        for idx, plan in enumerate(expert_plan):
            if plan.Priority != idx + 1:
                raise RoutingError("Priority ordering is inconsistent.")
            if plan.ExpertName != execution_order[idx]:
                raise RoutingError("ExpertPlan inconsistency with ExecutionOrder.")
            if plan.Weight != weights[plan.ExpertName]:
                raise RoutingError("ExpertPlan inconsistency with ExpertWeights.")
        
        processing_time = time.time() - start_time
        
        metadata["ConfidenceBuffer"] = confidence_buffer
        metadata["FallbackThreshold"] = fallback_threshold
        metadata["RoutingTimestamp"] = datetime.now().isoformat()
        metadata["RoutingStrategy"] = strategy
        metadata["LayerVersion"] = self.LayerVersion
        metadata["ProcessingTime"] = processing_time
        
        frozen_metadata = MappingProxyType(metadata)
        frozen_weights = MappingProxyType(weights)
        
        logger.info(f"Primary Expert: {primary_expert}")
        logger.info(f"Secondary Experts: {secondary_experts}")
        logger.info(f"Execution Order: {execution_order}")
        logger.info(f"Routing Strategy: {strategy}")
        logger.info(f"Requires Fusion: {requires_fusion}")
        logger.info(f"Routing Confidence: {routing_confidence:.4f}")
        logger.info(f"Weights: {weights}")
        logger.info(f"Expert Plan: {[p.ExpertName for p in expert_plan]}")
        logger.info(f"Processing Time: {processing_time:.4f} s")
        
        return RoutingDecision(
            Parent=characterization_report,
            ExpertPlan=expert_plan,
            PrimaryExpert=primary_expert,
            SecondaryExperts=secondary_experts,
            ExecutionOrder=execution_order,
            ExpertWeights=frozen_weights,
            RoutingStrategy=strategy,
            RoutingConfidence=routing_confidence,
            RequiresFusion=requires_fusion,
            FallbackMode=fallback_mode,
            Metadata=frozen_metadata,
            ProcessingTime=processing_time,
            LayerVersion=self.LayerVersion
        )

    def _validate_input(self, report: Any) -> None:
        if not isinstance(report, CharacterizationReport):
            raise RoutingError(f"Input object must be a CharacterizationReport, got {type(report).__name__}.")
            
        if not hasattr(report, "NoiseProbabilities") or not hasattr(report, "ActivatedCandidates"):
            raise RoutingError("Input object lacks required 'NoiseProbabilities' or 'ActivatedCandidates' attributes.")
