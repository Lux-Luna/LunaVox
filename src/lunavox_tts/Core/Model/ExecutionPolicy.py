"""
Execution Policy - Provider selection configuration for hybrid CPU/GPU modes.

Defines strategies for assigning execution providers to different model components,
enabling flexible deployment on CPU-only, GPU, or hybrid configurations.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class ExecutionMode(Enum):
    """Available execution modes."""
    CPU_ONLY = "cpu"
    GPU_ONLY = "gpu"
    HYBRID = "hybrid"  # CPU for frontend, GPU for inference


@dataclass
class ExecutionPolicy:
    """
    Configuration for model component execution providers.
    
    Enables per-component provider assignment for optimal resource utilization.
    The hybrid mode keeps BERT/HuBERT on CPU while running T2S/VITS on GPU.
    
    Attributes:
        mode: The overall execution mode
        component_providers: Override providers for specific components
    """
    mode: ExecutionMode = ExecutionMode.GPU_ONLY
    
    # Component-specific provider overrides
    # Key: component pattern (e.g., "T2S_*", "VITS", "BERT")
    # Value: List of providers in priority order
    component_providers: Dict[str, List[str]] = field(default_factory=dict)
    
    def get_providers_for(self, component_name: str) -> Optional[List[str]]:
        """
        Get providers for a specific component.
        
        Args:
            component_name: Name of the component (e.g., "T2S_ENCODER", "VITS")
            
        Returns:
            List of providers if explicitly configured, None to use default.
        """
        # Check exact match first
        if component_name in self.component_providers:
            return self.component_providers[component_name]
        
        # Check pattern matches (e.g., "T2S_*")
        for pattern, providers in self.component_providers.items():
            if pattern.endswith("*") and component_name.startswith(pattern[:-1]):
                return providers
        
        return None


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

HYBRID_POLICY = ExecutionPolicy(
    mode=ExecutionMode.HYBRID,
    component_providers={
        # Keep BERT on CPU to save VRAM
        "BERT": ["CPUExecutionProvider"],
        "HUBERT": ["CPUExecutionProvider"],
        # Run inference on GPU
        "T2S_*": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "VITS": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "PROMPT_ENCODER": ["CUDAExecutionProvider", "CPUExecutionProvider"],
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
        "hybrid": HYBRID_POLICY,
    }
    return policies.get(name.lower(), GPU_ACCELERATED_POLICY)
