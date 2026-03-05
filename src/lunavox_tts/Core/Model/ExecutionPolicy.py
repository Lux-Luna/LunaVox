"""
Execution Policy - Provider selection configuration for CPU/GPU modes.

Defines strategies for assigning execution providers to model components.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class EnvironmentMismatchError(Exception):
    """Raised when requested execution mode doesn't match environment capabilities."""
    pass


class ExecutionMode(Enum):
    """Available execution modes."""
    CPU_ONLY = "cpu"
    GPU_ONLY = "gpu"


@dataclass
class ExecutionPolicy:
    """
    Configuration for model component execution providers.
    
    Attributes:
        mode: The overall execution mode (cpu or gpu)
        component_providers: Provider list for all components
    """
    mode: ExecutionMode = ExecutionMode.GPU_ONLY
    
    # Provider list applied to all components
    component_providers: Dict[str, List[str]] = field(default_factory=dict)
    
    def get_providers_for(self, component_name: str) -> Optional[List[str]]:
        """Get providers for a component (returns default list)."""
        return self.component_providers.get("*")


# Pre-defined execution policies
CPU_ONLY_POLICY = ExecutionPolicy(
    mode=ExecutionMode.CPU_ONLY,
    component_providers={
        "*": ["CPUExecutionProvider"],
    }
)

GPU_ACCELERATED_POLICY = ExecutionPolicy(
    mode=ExecutionMode.GPU_ONLY,
    component_providers={
        "*": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }
)


def get_default_policy() -> ExecutionPolicy:
    """Get the default execution policy based on environment."""
    from ...Utils.EnvManager import env_manager
    mode = env_manager.get_mode()
    
    if mode == "cpu":
        return CPU_ONLY_POLICY
    else:
        return GPU_ACCELERATED_POLICY


def get_policy_by_name(name: str) -> ExecutionPolicy:
    """Get an execution policy by name."""
    policies = {
        "cpu": CPU_ONLY_POLICY,
        "gpu": GPU_ACCELERATED_POLICY,
    }
    return policies.get(name.lower(), GPU_ACCELERATED_POLICY)
