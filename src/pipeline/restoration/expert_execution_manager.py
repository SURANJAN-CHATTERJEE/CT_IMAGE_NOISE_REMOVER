import copy
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Any, Tuple, List, Dict, Optional

from src.pipeline.noise.noise_generator import NoisyVolume
from src.pipeline.routing.adaptive_router import RoutingDecision, ExpertExecution
from src.interfaces.expert import BaseExpert, ExpertResult

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Base exception for execution manager errors."""
    pass


class ExpertNotFoundError(ExecutionError):
    """Raised when a required expert is not registered or found."""
    pass


class ExpertExecutionError(ExecutionError):
    """Raised when an expert fails to execute properly."""
    pass


@dataclass(frozen=True)
class ExecutionResult:
    """
    Immutable representation of the overall expert execution results.
    """
    Parent: RoutingDecision
    ExecutedExperts: Tuple[str, ...]
    SkippedExperts: Tuple[str, ...]
    ExpertResults: Tuple[ExpertResult, ...]
    ExecutionOrder: Tuple[str, ...]
    ExecutionStatistics: Mapping[str, float]
    Metadata: Mapping[str, Any]
    ProcessingTime: float
    LayerVersion: str


class ExpertExecutionManager:
    """
    Engine for orchestrating and executing the expert routing plan.
    This layer does not perform denoising or fusion; it only orchestrates execution.
    """
    
    VERSION: str = "1.0.0"

    def __init__(self, expert_registry: Optional[Mapping[str, BaseExpert]] = None):
        """
        Initializes the execution manager with an optional expert registry.
        """
        self.registry = dict(expert_registry) if expert_registry else {}

    def register_expert(self, expert_name: str, expert_instance: BaseExpert) -> None:
        """
        Registers an expert instance into the execution manager.
        """
        self.registry[expert_name] = expert_instance

    def execute(self, routing_decision: Any, noisy_volume: Any) -> ExecutionResult:
        """
        Executes the routing decision plan on the noisy volume.
        
        Args:
            routing_decision: The RoutingDecision object from Layer 8.
            noisy_volume: The NoisyVolume object from Layer 6.
            
        Returns:
            ExecutionResult: Immutable dataclass containing execution results.
            
        Raises:
            ExecutionError: For validation and orchestration failures.
            ExpertNotFoundError: If a required expert is not registered.
            ExpertExecutionError: If an expert fails during execution.
        """
        start_time = time.time()
        
        self._validate_inputs(routing_decision, noisy_volume)
        
        metadata = copy.deepcopy(dict(routing_decision.Metadata))
        
        expert_plan = routing_decision.ExpertPlan
        expected_order = routing_decision.ExecutionOrder
        
        self._validate_execution_order(expert_plan, expected_order)
        
        executed = []
        skipped = []
        results = []
        exec_times = {}
        
        for plan in expert_plan:
            if not plan.Execute:
                skipped.append(plan.ExpertName)
                logger.info(f"Skipping expert {plan.ExpertName} (Execute=False)")
                continue
                
            expert_name = plan.ExpertName
            if expert_name not in self.registry:
                raise ExpertNotFoundError(f"Expert {expert_name} is required but not registered.")
                
            expert = self.registry[expert_name]
            
            logger.info(f"Executing expert {expert_name} (Priority={plan.Priority}, Weight={plan.Weight})")
            
            try:
                exec_start = time.time()
                result = expert.execute(noisy_volume, plan)
                exec_time = time.time() - exec_start
                
                if not isinstance(result, ExpertResult):
                    raise ExpertExecutionError(f"Expert {expert_name} did not return an ExpertResult.")
                    
                if result.ExpertName != expert_name:
                    raise ExpertExecutionError(f"ExpertResult name {result.ExpertName} does not match {expert_name}.")
                    
                results.append(result)
                executed.append(expert_name)
                exec_times[expert_name] = exec_time
                logger.info(f"Expert {expert_name} finished in {exec_time:.4f}s")
                
            except Exception as e:
                logger.error(f"Execution failed for expert {expert_name}: {str(e)}")
                raise ExpertExecutionError(f"Execution failed for {expert_name}: {str(e)}") from e
                
        if not executed:
            raise ExecutionError("No experts were executed. All experts were skipped or none were provided.")
            
        if len(set(executed)) != len(executed):
            raise ExecutionError("Duplicate expert executions detected.")
            
        total_processing_time = time.time() - start_time
        
        exec_stats = {
            "TotalProcessingTime": total_processing_time,
            "ExecutedCount": float(len(executed)),
            "SkippedCount": float(len(skipped)),
            "AverageExpertTime": sum(exec_times.values()) / len(executed) if executed else 0.0
        }
        exec_stats.update({f"{k}_Time": v for k, v in exec_times.items()})
        
        metadata["ExecutionTimestamp"] = datetime.now().isoformat()
        metadata["LayerVersion"] = self.VERSION
        metadata["ProcessingTime"] = total_processing_time
        
        frozen_metadata = MappingProxyType(metadata)
        frozen_stats = MappingProxyType(exec_stats)
        
        logger.info(f"Executed Experts: {executed}")
        logger.info(f"Skipped Experts: {skipped}")
        logger.info(f"Total Execution Time: {total_processing_time:.4f} s")
        
        return ExecutionResult(
            Parent=routing_decision,
            ExecutedExperts=tuple(executed),
            SkippedExperts=tuple(skipped),
            ExpertResults=tuple(results),
            ExecutionOrder=expected_order,
            ExecutionStatistics=frozen_stats,
            Metadata=frozen_metadata,
            ProcessingTime=total_processing_time,
            LayerVersion=self.VERSION
        )

    def _validate_inputs(self, routing_decision: Any, noisy_volume: Any) -> None:
        if not isinstance(routing_decision, RoutingDecision):
            raise ExecutionError(f"Invalid routing_decision type: {type(routing_decision).__name__}")
        if not isinstance(noisy_volume, NoisyVolume):
            raise ExecutionError(f"Invalid noisy_volume type: {type(noisy_volume).__name__}")
            
    def _validate_execution_order(self, expert_plan: Tuple[ExpertExecution, ...], expected_order: Tuple[str, ...]) -> None:
        if len(expert_plan) != len(expected_order):
            raise ExecutionError("Expert plan length does not match execution order length.")
            
        for i, plan in enumerate(expert_plan):
            if plan.ExpertName != expected_order[i]:
                raise ExecutionError(f"Execution order inconsistency at index {i}: Expected {expected_order[i]}, got {plan.ExpertName}.")
