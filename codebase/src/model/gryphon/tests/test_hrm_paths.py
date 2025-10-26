import sys
import os
sys.path.append('/home/ravkeave/v1/codebase')

import jax
import jax.numpy as jnp
import pytest

from src.model.gryphon.gryphon_hrm_model import GryphonHRMModel, create_attention_mask, create_position_ids
from src.model.gryphon.gryphon_config import get_gryphon_small_config


def make_small_config(hrm_enabled: bool, use_act: bool):
    config = get_gryphon_small_config()
    config.max_sequence_length = 256
    config.block_size = 64
    config.num_global_blocks = 2
    config.window_size = 3
    config.num_random_blocks = 2
    config.hrm_enabled = hrm_enabled
    config.hrm_use_act = use_act
    config.hrm_max_steps = 3
    config.use_mixed_precision = False
    config.use_gradient_checkpointing = False
    config.__post_init__()
    return config


def make_dummy_batch(config, batch_size=2):
    rng = jax.random.PRNGKey(0)
    seq_len = config.max_sequence_length
    vocab_size = getattr(config, 'vocab_size', 50257)
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, vocab_size)
    attn_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    pos_ids = create_position_ids(input_ids)
    return rng, input_ids, attn_mask, pos_ids


def assert_state_shapes(new_state, config, batch_size):
    assert new_state.s5_state.shape == (batch_size, config.s5_state_dim)
    assert new_state.global_tokens.shape == (batch_size, config.num_global_blocks, config.d_model)
    # HRM carry always present; if HRM disabled, it's a zero dummy of correct shape
    assert new_state.hrm_carry.z_H.shape == (batch_size, config.max_sequence_length, config.d_model)
    assert new_state.hrm_carry.z_L.shape == (batch_size, config.max_sequence_length, config.d_model)


@pytest.mark.parametrize("hrm_enabled,use_act", [
    (False, False),
    (True, False),
    (True, True),
])
def test_hrm_paths_forward(hrm_enabled, use_act):
    config = make_small_config(hrm_enabled=hrm_enabled, use_act=use_act)
    model = GryphonHRMModel(config=config)

    rng, input_ids, attn_mask, pos_ids = make_dummy_batch(config)

    batch_size = input_ids.shape[0]

    # Initialize variables (params and others)
    variables = model.init(rng, input_ids, None, attn_mask, pos_ids, True, None)

    # Capture initial global tokens using apply(method=init_state) so setup-defined fields are accessible
    init_state = model.apply(variables, method=model.init_state, batch_size=batch_size, seq_len=config.max_sequence_length)
    init_global = init_state.global_tokens

    # Run forward with deterministic=True to avoid dropout RNG requirements
    logits, new_state = model.apply(variables, input_ids, None, attn_mask, pos_ids, True, None)

    assert logits.shape == (batch_size, config.max_sequence_length, config.vocab_size)
    assert_state_shapes(new_state, config, batch_size)

    # If HRM enabled, global tokens should be updated by attention integration
    if hrm_enabled:
        assert not jnp.allclose(new_state.global_tokens, init_global), "Global tokens should update via attention"


if __name__ == "__main__":
    # Allow running this test module directly for quick validation
    test_hrm_paths_forward(False, False)
    test_hrm_paths_forward(True, False)
    test_hrm_paths_forward(True, True)
    print("All HRM path tests passed.")