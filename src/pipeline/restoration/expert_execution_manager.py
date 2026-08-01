import copy
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Any, Tuple, List, Dict, Optional

import numpy as np

from src.pipeline.noise.noise_generator import NoisyVolume
from src.pipeline.routing.adaptive_router import RoutingDecision, ExpertExecution
from src.interfaces.expert import BaseExpert, ExpertResult
from src.core.registry import ExpertRegistry
from src.core.types import ExpertStatus

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Base exception for execution manager errors."""
    pass


@dataclass(frozen=True)
class ExecutionResult:
    """Immutable representation of the overall expert execution results."""
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
    """Engine for orchestrating and executing the expert routing plan."""
    
    LayerVersion: str = "v1.0.1"

    def __init__(self, expert_registry: Optional[Any] = None):
        self.registry = ExpertRegistry

    def execute(self, routing_decision: Any, noisy_volume: Any) -> ExecutionResult:
        start_time = time.time()
        
        self._validate_inputs(routing_decision, noisy_volume)
        metadata = copy.deepcopy(dict(routing_decision.Metadata))
        
        expert_plan = routing_decision.ExpertPlan
        expected_order = routing_decision.ExecutionOrder
        
        executable_experts = [p for p in expert_plan if p.Execute]
        if not executable_experts:
            raise ExecutionError("No executable experts remain in the plan.")
            
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
            if not self.registry.exists(expert_name):
                logger.error(f"Expert {expert_name} is required but not registered.")
                skipped.append(expert_name)
                continue
                
            expert = self.registry.get(expert_name)
            
            logger.info(f"Executing expert {expert_name} (Priority={plan.Priority}, Weight={plan.Weight})")
            
            exec_start = time.time()
            try:
                result = expert.execute(noisy_volume, plan)
                exec_time = time.time() - exec_start
                
                if not isinstance(result, ExpertResult):
                    raise ExecutionError(f"Expert {expert_name} did not return an ExpertResult.")
                    
                results.append(result)
                executed.append(expert_name)
                exec_times[expert_name] = exec_time
                logger.info(f"Expert {expert_name} finished in {exec_time:.4f}s")
                
            except Exception as e:
                exec_time = time.time() - exec_start
                logger.error(f"Execution failed for expert {expert_name}: {str(e)}")
                
                fail_metadata = {
                    "ExceptionType": type(e).__name__,
                    "ExceptionMessage": str(e),
                    "ExecutionTime": exec_time
                }
                
                # Mock result for FAILED
                dummy_output = np.zeros_like(noisy_volume.Volume) if hasattr(noisy_volume, "Volume") else np.zeros((1,1,1))
                fail_result = ExpertResult(
                    ExpertName=expert_name,
                    InputVolume=noisy_volume,
                    OutputVolume=dummy_output,
                    Confidence=plan.Confidence,
                    Weight=0.0,
                    ExecutionTime=exec_time,
                    Status=ExpertStatus.FAILED,
                    Statistics={},
                    Metadata=MappingProxyType(fail_metadata),
                    LayerVersion=self.LayerVersion
                )
                results.append(fail_result)
                
        # Raise ONLY if all fail or no executable
        if not executed and not results:
            raise ExecutionError("All experts failed or none were executable.")
            
        total_processing_time = time.time() - start_time
        
        exec_stats = {
            "TotalProcessingTime": total_processing_time,
            "ExecutedCount": float(len(executed)),
            "SkippedCount": float(len(skipped)),
            "AverageExpertTime": sum(exec_times.values()) / max(len(exec_times), 1)
        }
        exec_stats.update({f"{k}_Time": v for k, v in exec_times.items()})
        
        metadata["ExecutionTimestamp"] = datetime.now().isoformat()
        metadata["LayerVersion"] = self.LayerVersion
        metadata["ProcessingTime"] = total_processing_time
        
        return ExecutionResult(
            Parent=routing_decision,
            ExecutedExperts=tuple(executed),
            SkippedExperts=tuple(skipped),
            ExpertResults=tuple(results),
            ExecutionOrder=expected_order,
            ExecutionStatistics=MappingProxyType(exec_stats),
            Metadata=MappingProxyType(metadata),
            ProcessingTime=total_processing_time,
            LayerVersion=self.LayerVersion
        )

    def _validate_inputs(self, routing_decision: Any, noisy_volume: Any) -> None:
        if not isinstance(routing_decision, RoutingDecision):
            raise ExecutionError(f"Invalid routing_decision type: {type(routing_decision).__name__}")
        if not isinstance(noisy_volume, NoisyVolume):
            raise ExecutionError(f"Invalid noisy_volume type: {type(noisy_volume).__name__}")
