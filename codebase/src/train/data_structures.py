from typing import Any, Optional
import flax.struct

# Define a custom PRNG key type for clarity
PRNGKey = Any

@flax.struct.dataclass
class TrainingState:
    """Main training state for the Valkyrie model."""
    params: Any
    opt_state: Any
    step: int
    rng: PRNGKey
    s5_states: Any
    chunk_position: int
    phase_index: int
    hrm_enabled: bool
    hrm_training_state: Optional[Any] = None