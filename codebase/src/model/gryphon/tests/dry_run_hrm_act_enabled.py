import sys
import os
sys.path.append('/home/ravkeave/v1/codebase')

import jax
import jax.numpy as jnp

from src.model.gryphon.gryphon_hrm_model import GryphonHRMModel, create_attention_mask, create_position_ids
from src.model.gryphon.gryphon_config import GryphonConfig, get_gryphon_small_config


def main():
    # Create a minimal config with hrm_enabled=True and ACT enabled
    config = get_gryphon_small_config()
    config.max_sequence_length = 256
    config.block_size = 64
    config.num_global_blocks = 2
    config.window_size = 3
    config.num_random_blocks = 2
    config.hrm_enabled = True
    config.hrm_use_act = True  # enable ACT path
    config.hrm_max_steps = 3   # keep ACT steps small for dry-run
    config.use_mixed_precision = False
    config.use_gradient_checkpointing = False
    # Ensure config is re-validated after overrides
    config.__post_init__()

    # Create model
    model = GryphonHRMModel(config=config)

    # Dummy inputs
    rng = jax.random.PRNGKey(0)
    batch_size = 2
    seq_len = config.max_sequence_length
    vocab_size = config.vocab_size if hasattr(config, 'vocab_size') else 5000
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, vocab_size)

    # Optional masks
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    position_ids = create_position_ids(input_ids)

    # Initialize parameters (deterministic=True to avoid dropout RNG)
    params = model.init(rng, input_ids, None, attention_mask, position_ids, True, None)

    # Forward pass with HRM ACT enabled
    logits, new_state = model.apply(params, input_ids, None, attention_mask, position_ids, True, None)

    print("=== HRM ACT-enabled dry-run completed ===")
    print("logits:", logits.shape)
    print("s5_state:", new_state.s5_state.shape)
    print("global_tokens:", new_state.global_tokens.shape)
    # HRM carry shapes
    print("hrm_carry.z_H:", new_state.hrm_carry.z_H.shape)
    print("hrm_carry.z_L:", new_state.hrm_carry.z_L.shape)


if __name__ == "__main__":
    main()