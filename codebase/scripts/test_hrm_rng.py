import sys
sys.path.append('/home/ravkeave/v1/codebase')

import jax
import jax.numpy as jnp

from src.model.gryphon.gryphon_hrm_model import GryphonHRMModel, create_attention_mask, create_position_ids
from src.model.gryphon.gryphon_config import get_gryphon_small_config
from src.train.hrm_training_loop import HRMTrainingLoop, HRMTrainingConfig, HRMBatch


def main():
    # Configure a small Gryphon HRM model for a quick RNG pass check
    config = get_gryphon_small_config()
    config.max_sequence_length = 256
    config.block_size = 64
    config.num_global_blocks = 2
    config.window_size = 3
    config.num_random_blocks = 2
    config.hrm_enabled = True
    config.hrm_use_act = False
    config.use_mixed_precision = False
    config.use_gradient_checkpointing = False
    config.__post_init__()

    # Construct model
    model = GryphonHRMModel(config=config)

    # Dummy batch
    rng = jax.random.PRNGKey(0)
    batch_size = 2
    seq_len = config.max_sequence_length
    vocab_size = getattr(config, 'vocab_size', 5000)
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, vocab_size)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    position_ids = create_position_ids(input_ids)

    # Initialize parameters and state
    variables = model.init(rng, input_ids, None, attention_mask, position_ids, True, None)
    params = variables['params']
    # IMPORTANT: call init_state via apply(method=...) so submodules are accessible
    init_state = model.apply(variables, batch_size, seq_len, method=model.init_state)

    # Build HRM batch
    hrm_batch = HRMBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        doc_ids=jnp.zeros_like(input_ids),
        segment_ids=jnp.zeros_like(input_ids),
        linked_next=jnp.zeros((batch_size,), dtype=bool),
        source_ids=jnp.zeros((batch_size,))
    )

    # Create training loop config and loop
    hrm_config = HRMTrainingConfig()
    loop = HRMTrainingLoop(model=model, config=hrm_config)

    # Run HRM forward with training=True to require dropout/random RNGs
    step_rng = jax.random.PRNGKey(42)
    logits, new_state, metrics = loop._hrm_forward_with_one_step_gradient(
        params=params,
        model_state=init_state,
        batch=hrm_batch,
        rng_key=step_rng,
        training=True
    )

    print("=== HRM RNG pass check completed ===")
    print("logits shape:", logits.shape)
    print("s5_state shape:", new_state.s5_state.shape)
    print("global_tokens shape:", new_state.global_tokens.shape)
    print("metrics:", metrics)


if __name__ == "__main__":
    main()