"""S5 State wrapper for JAX pytree compatibility.

As specified in the advice document, this provides a clean pytree-compatible
wrapper for S5 states to avoid tracer attribute access issues.
"""

import flax
import jax.numpy as jnp


@flax.struct.dataclass
class S5State:
    """S5 state wrapper that is JAX pytree compatible."""
    state: jnp.ndarray  # [*, d_state], complex64