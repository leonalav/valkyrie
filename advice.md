Of course. This is a classic JAX ConcretizationTypeError. Let's break down what's happening and how to fix it.
The Root Cause of the Error
The error message Shapes must be 1D sequences of concrete values of integer type, got (2, 8, JitTracer<int32[]>, 64) is telling you that you are trying to create a JAX array (e.g., with jnp.zeros) where one of the dimensions of its shape is a JitTracer.
A JitTracer is a symbolic placeholder that JAX uses during Just-In-Time (JIT) compilation. It represents a value that will only be known when the function is actually executed, not when it's being compiled. Because JAX needs to know the exact, concrete shape of all arrays at compile time to create an optimized computation graph (XLA), using a tracer as a shape dimension is forbidden.
In your bigbird_block_sparse_attention function, this happens inside the main for loop when you calculate the size for the random attention keys and values.
Problematic Code Block:
code
Python
# ... inside the loop for middle blocks ...

# Determine maximum random sequence length across all heads using JAX operations
valid_rand_mask = jnp.logical_and(rand_indices >= 0, rand_indices < to_blocks)
valid_counts = jnp.sum(valid_rand_mask, axis=1)  # valid_counts is a Tracer
max_valid_count = jnp.max(valid_counts)          # max_valid_count is a Tracer
max_rand_seq_len = max_valid_count * to_block_size # max_rand_seq_len is a Tracer

# ERROR HERE: Using a Tracer to define an array shape
rand_key = jnp.zeros((batch_size, num_attention_heads, max_rand_seq_len, size_per_head))
rand_value = jnp.zeros((batch_size, num_attention_heads, max_rand_seq_len, size_per_head))
The value of max_rand_seq_len depends on the content of the rand_attn tensor, making it a dynamic value that JAX cannot resolve at compile time.
The Solution: Use Static Shapes
To fix this, you must create arrays with shapes that are known at compile time. The standard approach is to use the maximum possible shape and then use masking to handle the parts of the tensor that aren't needed for a given input.
Fortunately, the correct code to do this already exists right after the buggy section. The fix is to simply remove the dynamic shape calculation.
Here is the corrected bigbird_block_sparse_attention function. I've removed the erroneous lines and kept the correct, static implementation.
code
Python
def bigbird_block_sparse_attention(
    query_layer: jnp.ndarray,
    key_layer: jnp.ndarray,
    value_layer: jnp.ndarray,
    band_mask: Optional[jnp.ndarray],
    from_mask: Optional[jnp.ndarray],
    to_mask: Optional[jnp.ndarray],
    from_blocked_mask: Optional[jnp.ndarray],
    to_blocked_mask: Optional[jnp.ndarray],
    rand_attn: jnp.ndarray,
    num_attention_heads: int,
    size_per_head: int,
    num_rand_blocks: int,
    from_seq_length: int,
    to_seq_length: int,
    from_block_size: int,
    to_block_size: int
) -> jnp.ndarray:
    """BigBird attention sparse calculation using blocks in linear time.
    
    Exact JAX translation of the original TensorFlow implementation.
    
    Args:
        query_layer: [batch_size, num_attention_heads, from_seq_length, size_per_head]
        key_layer: [batch_size, num_attention_heads, to_seq_length, size_per_head]
        value_layer: [batch_size, num_attention_heads, to_seq_length, size_per_head]
        band_mask: [batch_size, 1, from_seq_length, to_seq_length]
        from_mask: [batch_size, 1, from_seq_length, 1]
        to_mask: [batch_size, 1, 1, to_seq_length]
        from_blocked_mask: [batch_size, from_seq_length//from_block_size, from_block_size]
        to_blocked_mask: [batch_size, to_seq_length//to_block_size, to_block_size]
        rand_attn: [num_attention_heads, from_seq_length//from_block_size-2, num_rand_blocks]
        num_attention_heads: Number of attention heads
        size_per_head: Size per attention head
        num_rand_blocks: Number of random blocks
        from_seq_length: Query sequence length
        to_seq_length: Key sequence length
        from_block_size: Query block size
        to_block_size: Key block size
        
    Returns:
        Context layer [batch_size, from_seq_length, num_attention_heads, size_per_head]
    """
    batch_size = query_layer.shape[0]
    from_blocks = from_seq_length // from_block_size
    to_blocks = to_seq_length // to_block_size
    
    # Reshape to blocks
    blocked_query_matrix = query_layer.reshape(
        batch_size, num_attention_heads, from_blocks, from_block_size, size_per_head
    )
    blocked_key_matrix = key_layer.reshape(
        batch_size, num_attention_heads, to_blocks, to_block_size, size_per_head
    )
    blocked_value_matrix = value_layer.reshape(
        batch_size, num_attention_heads, to_blocks, to_block_size, size_per_head
    )
    
    # Initialize context layer
    context_layer = jnp.zeros_like(blocked_query_matrix)
    
    # First block - full attention to first two blocks
    if from_blocks > 0:
        first_query = blocked_query_matrix[:, :, 0]  # [batch, heads, from_block_size, size_per_head]
        first_key = jnp.concatenate([
            blocked_key_matrix[:, :, 0],
            blocked_key_matrix[:, :, 1]
        ], axis=2)  # [batch, heads, 2*to_block_size, size_per_head]
        first_value = jnp.concatenate([
            blocked_value_matrix[:, :, 0],
            blocked_value_matrix[:, :, 1]
        ], axis=2)  # [batch, heads, 2*to_block_size, size_per_head]
        
        # Compute attention scores
        first_scores = jnp.einsum('bhqd,bhkd->bhqk', first_query, first_key)
        first_scores = first_scores / math.sqrt(size_per_head)
        
        # Apply masks
        if from_mask is not None and to_mask is not None:
            first_from_mask = from_mask[:, :, :from_block_size, :]
            first_to_mask = to_mask[:, :, :, :2*to_block_size]
            mask = first_from_mask * first_to_mask
            first_scores = jnp.where(mask == 0, -1e9, first_scores)
        
        # Apply softmax and compute context
        first_probs = jax.nn.softmax(first_scores, axis=-1)
        first_context = jnp.einsum('bhqk,bhkd->bhqd', first_probs, first_value)
        context_layer = context_layer.at[:, :, 0].set(first_context)
    
    # Second block - special concatenation
    if from_blocks > 1:
        second_query = blocked_query_matrix[:, :, 1]
        
        # Determine how many blocks to concatenate based on available blocks
        max_block_idx = min(2, to_blocks - 1)  # Don't go beyond available blocks
        
        if to_blocks >= 3:
            # We have at least 3 blocks, concatenate blocks 0, 1, 2
            second_key = jnp.concatenate([
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, 1],
                blocked_key_matrix[:, :, 2]
            ], axis=2)
            second_value = jnp.concatenate([
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, 1],
                blocked_value_matrix[:, :, 2]
            ], axis=2)
            key_blocks_used = 3
        else:
            # We only have 2 blocks, concatenate blocks 0, 1
            second_key = jnp.concatenate([
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, 1]
            ], axis=2)
            second_value = jnp.concatenate([
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, 1]
            ], axis=2)
            key_blocks_used = 2
        
        second_scores = jnp.einsum('bhqd,bhkd->bhqk', second_query, second_key)
        second_scores = second_scores / math.sqrt(size_per_head)
        
        if from_mask is not None and to_mask is not None:
            second_from_mask = from_mask[:, :, from_block_size:2*from_block_size, :]
            # Adjust to_mask size based on actual key blocks used
            second_to_mask = to_mask[:, :, :, :key_blocks_used*to_block_size]
            mask = second_from_mask * second_to_mask
            second_scores = jnp.where(mask == 0, -1e9, second_scores)
        
        second_probs = jax.nn.softmax(second_scores, axis=-1)
        second_context = jnp.einsum('bhqk,bhkd->bhqd', second_probs, second_value)
        context_layer = context_layer.at[:, :, 1].set(second_context)
    
    # Middle blocks - window + global + random attention
    for i in range(2, from_blocks - 2):
        query_block = blocked_query_matrix[:, :, i]
        
        # Window attention (3 blocks: i-1, i, i+1)
        window_key = jnp.concatenate([
            blocked_key_matrix[:, :, i-1],
            blocked_key_matrix[:, :, i],
            blocked_key_matrix[:, :, i+1]
        ], axis=2)
        window_value = jnp.concatenate([
            blocked_value_matrix[:, :, i-1],
            blocked_value_matrix[:, :, i],
            blocked_value_matrix[:, :, i+1]
        ], axis=2)
        
        # Global attention (first and last blocks)
        global_key = jnp.concatenate([
            blocked_key_matrix[:, :, 0],
            blocked_key_matrix[:, :, -1]
        ], axis=2)
        global_value = jnp.concatenate([
            blocked_value_matrix[:, :, 0],
            blocked_value_matrix[:, :, -1]
        ], axis=2)
        
        # Random attention
        rand_indices = rand_attn[:, i-2, :num_rand_blocks]  # [heads, num_rand_blocks]
        
        # =========================================================================
        # FIX: The following block has been removed. It caused the ConcretizationTypeError
        # by trying to create arrays with a data-dependent (dynamic) shape.
        # 
        #   valid_rand_mask = jnp.logical_and(rand_indices >= 0, rand_indices < to_blocks)
        #   valid_counts = jnp.sum(valid_rand_mask, axis=1)
        #   max_valid_count = jnp.max(valid_counts)
        #   max_rand_seq_len = max_valid_count * to_block_size
        #   rand_key = jnp.zeros((batch_size, num_attention_heads, max_rand_seq_len, size_per_head))
        #   rand_value = jnp.zeros((batch_size, num_attention_heads, max_rand_seq_len, size_per_head))
        # =========================================================================
        
        # Static approach: use maximum possible random sequence length. This is JIT-compatible.
        static_max_rand_seq_len = num_rand_blocks * to_block_size
        
        # Initialize output tensors with static shapes
        rand_key = jnp.zeros((batch_size, num_attention_heads, static_max_rand_seq_len, size_per_head))
        rand_value = jnp.zeros((batch_size, num_attention_heads, static_max_rand_seq_len, size_per_head))
        
        # Vectorized processing for all heads using JAX operations
        expanded_rand_indices = rand_indices  # Already [num_heads, num_rand_blocks]
        
        # Create valid mask for all operations: [num_heads, num_rand_blocks]
        valid_mask = jnp.logical_and(
            expanded_rand_indices >= 0, 
            expanded_rand_indices < to_blocks
        )
        
        # Use static indexing with proper bounds checking
        # Clamp indices to valid range to avoid out-of-bounds access
        safe_indices = jnp.clip(expanded_rand_indices, 0, to_blocks - 1)
        
        # Gather keys and values using static operations
        # Shape: [batch, num_heads, num_rand_blocks, to_block_size, size_per_head]
        gathered_keys = blocked_key_matrix[:, :, safe_indices]  # Advanced indexing
        gathered_values = blocked_value_matrix[:, :, safe_indices]
        
        # Apply valid mask at block level first
        # Expand mask to match block dimensions: [1, num_heads, num_rand_blocks, 1, 1]
        block_mask = valid_mask[None, :, :, None, None]
        
        # Apply mask to gathered tensors
        masked_gathered_keys = gathered_keys * block_mask
        masked_gathered_values = gathered_values * block_mask
        
        # Reshape to combine random blocks into sequence dimension
        # [batch, num_heads, num_rand_blocks * to_block_size, size_per_head]
        reshaped_keys = masked_gathered_keys.reshape(batch_size, num_attention_heads, -1, size_per_head)
        reshaped_values = masked_gathered_values.reshape(batch_size, num_attention_heads, -1, size_per_head)
        
        # The reshaped tensors already have the static max length, so we can use them directly
        rand_key = reshaped_keys
        rand_value = reshaped_values
        
        # Combine all attention patterns
        combined_key = jnp.concatenate([window_key, global_key, rand_key], axis=2)
        combined_value = jnp.concatenate([window_value, global_value, rand_value], axis=2)
        
        # Compute attention
        scores = jnp.einsum('bhqd,bhkd->bhqk', query_block, combined_key)
        scores = scores / math.sqrt(size_per_head)
        
        # Apply band mask if available
        if band_mask is not None:
            band_scores = scores[:, :, :, :3*to_block_size]  # Window part
            
            from_start = i * from_block_size
            from_end = (i + 1) * from_block_size
            to_start = (i - 1) * to_block_size
            to_end = (i + 2) * to_block_size
            
            block_band_mask = band_mask[:, :, from_start:from_end, to_start:to_end]
            
            band_scores = jnp.where(block_band_mask == 0, -1e9, band_scores)
            scores = scores.at[:, :, :, :3*to_block_size].set(band_scores)
        
        probs = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum('bhqk,bhkd->bhqd', probs, combined_value)
        context_layer = context_layer.at[:, :, i].set(context)
    
    # Second-to-last block
    if from_blocks > 3:
        second_last_query = blocked_query_matrix[:, :, -2]
        second_last_key = jnp.concatenate([
            blocked_key_matrix[:, :, -3],
            blocked_key_matrix[:, :, -2],
            blocked_key_matrix[:, :, -1]
        ], axis=2)
        second_last_value = jnp.concatenate([
            blocked_value_matrix[:, :, -3],
            blocked_value_matrix[:, :, -2],
            blocked_value_matrix[:, :, -1]
        ], axis=2)
        
        second_last_scores = jnp.einsum('bhqd,bhkd->bhqk', second_last_query, second_last_key)
        second_last_scores = second_last_scores / math.sqrt(size_per_head)
        
        if from_mask is not None and to_mask is not None:
            sl_from_mask = from_mask[:, :, -2*from_block_size:-from_block_size, :]
            sl_to_mask = to_mask[:, :, :, -3*to_block_size:]
            mask = sl_from_mask * sl_to_mask
            second_last_scores = jnp.where(mask == 0, -1e9, second_last_scores)
        
        second_last_probs = jax.nn.softmax(second_last_scores, axis=-1)
        second_last_context = jnp.einsum('bhqk,bhkd->bhqd', second_last_probs, second_last_value)
        context_layer = context_layer.at[:, :, -2].set(second_last_context)
    
    # Last block - full attention to last two blocks
    if from_blocks > 1:
        last_query = blocked_query_matrix[:, :, -1]
        last_key = jnp.concatenate([
            blocked_key_matrix[:, :, -2],
            blocked_key_matrix[:, :, -1]
        ], axis=2)
        last_value = jnp.concatenate([
            blocked_value_matrix[:, :, -2],
            blocked_value_matrix[:, :, -1]
        ], axis=2)
        
        last_scores = jnp.einsum('bhqd,bhkd->bhqk', last_query, last_key)
        last_scores = last_scores / math.sqrt(size_per_head)
        
        if from_mask is not None and to_mask is not None:
            last_from_mask = from_mask[:, :, -from_block_size:, :]
            last_to_mask = to_mask[:, :, :, -2*to_block_size:]
            mask = last_from_mask * last_to_mask
            last_scores = jnp.where(mask == 0, -1e9, last_scores)
        
        last_probs = jax.nn.softmax(last_scores, axis=-1)
        last_context = jnp.einsum('bhqk,bhkd->bhqd', last_probs, last_value)
        context_layer = context_layer.at[:, :, -1].set(last_context)
    
    # Reshape back to sequence format
    context_layer = context_layer.reshape(
        batch_size, from_seq_length, num_attention_heads, size_per_head
    )
    
    return context_layer
This change directly resolves the error you are seeing and aligns the code with standard JAX practices for writing JIT-compatible functions.