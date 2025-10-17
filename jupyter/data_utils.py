# In data_utils.py

import numpy as np
from typing import List, Dict

def numpy_collate_fn(batch: List[Dict], tokenizer, max_length: int) -> Dict[str, np.ndarray]:
    """
    Collate function that returns standard NumPy arrays.
    This function should NOT use JAX to avoid GPU contention in worker processes.
    """
    if not batch:
        return {}

    input_ids = [item["input_ids"] for item in batch if "input_ids" in item]
    if not input_ids:
        return {}
    
    # Pad to the max length in the batch
    max_len_batch = max(len(ids) for ids in input_ids)
    padded_input_ids = np.full((len(input_ids), max_len_batch), tokenizer.pad_token_id, dtype=np.int32)
    
    for i, ids in enumerate(input_ids):
        padded_input_ids[i, :len(ids)] = ids
        
    labels = padded_input_ids.copy()
    labels[labels == tokenizer.pad_token_id] = -100  # Standard ignore index

    # Return pure NumPy arrays
    return {
        "input_ids": padded_input_ids,
        "labels": labels
    }