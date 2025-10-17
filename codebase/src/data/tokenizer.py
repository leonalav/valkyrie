"""Tokenizer configuration and utilities.

Implements GPT-2 tokenizer setup and configuration for FineWeb dataset processing.
Handles special tokens, padding, and efficient tokenization for long sequences.
"""

import jax
import jax.numpy as jnp
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer setup."""
    tokenizer_name: str = "gpt2"
    vocab_size: int = 50257
    max_length: int = 32768
    padding_side: str = "right"
    truncation_side: str = "right"
    
    # Special tokens
    pad_token: str = "<|endoftext|>"
    eos_token: str = "<|endoftext|>"
    bos_token: Optional[str] = None
    unk_token: str = "<|endoftext|>"
    
    # Processing options
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    return_token_type_ids: bool = False


def create_tokenizer(config: TokenizerConfig) -> PreTrainedTokenizerFast:
    """
    Create and configure GPT-2 tokenizer.
    
    Args:
        config: Tokenizer configuration
        
    Returns:
        Configured tokenizer
    """
    
    logger.info(f"Creating tokenizer: {config.tokenizer_name}")
    
    # Load base tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(config.tokenizer_name)
    
    # Configure special tokens
    special_tokens = {}
    
    if config.pad_token and tokenizer.pad_token is None:
        special_tokens['pad_token'] = config.pad_token
    
    if config.eos_token and tokenizer.eos_token != config.eos_token:
        special_tokens['eos_token'] = config.eos_token
    
    if config.bos_token and tokenizer.bos_token != config.bos_token:
        special_tokens['bos_token'] = config.bos_token
    
    if config.unk_token and tokenizer.unk_token != config.unk_token:
        special_tokens['unk_token'] = config.unk_token
    
    # Add special tokens if needed
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        logger.info(f"Added special tokens: {special_tokens}")
    
    # Configure tokenizer properties
    tokenizer.padding_side = config.padding_side
    tokenizer.truncation_side = config.truncation_side
    tokenizer.model_max_length = config.max_length
    
    # Verify vocab size
    actual_vocab_size = len(tokenizer)
    if actual_vocab_size != config.vocab_size:
        logger.warning(
            f"Vocab size mismatch: expected {config.vocab_size}, "
            f"got {actual_vocab_size}"
        )
    
    logger.info(f"Tokenizer configured:")
    logger.info(f"  Vocab size: {len(tokenizer)}")
    logger.info(f"  Max length: {tokenizer.model_max_length}")
    logger.info(f"  Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    logger.info(f"  EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    logger.info(f"  BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    
    return tokenizer


def tokenize_text(
    text: Union[str, List[str]],
    tokenizer: PreTrainedTokenizerFast,
    max_length: Optional[int] = None,
    padding: Union[bool, str] = False,
    truncation: bool = True,
    return_tensors: str = "np",
) -> Dict[str, Any]:
    """
    Tokenize text with proper configuration.
    
    Args:
        text: Input text or list of texts
        tokenizer: Configured tokenizer
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate
        return_tensors: Format for returned tensors
        
    Returns:
        Tokenized outputs
    """
    
    if max_length is None:
        max_length = tokenizer.model_max_length
    
    # Tokenize
    outputs = tokenizer(
        text,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
        return_attention_mask=True,
        add_special_tokens=True,
    )
    
    return outputs


def tokenize_for_training(
    texts: List[str],
    tokenizer: PreTrainedTokenizerFast,
    chunk_size: int = 8192,
    overlap_size: int = 512,
    min_chunk_size: int = 1024,
) -> List[Dict[str, jnp.ndarray]]:
    """
    Tokenize texts for training with chunking and overlap.
    
    This function handles long documents by:
    1. Tokenizing the full document
    2. Splitting into overlapping chunks
    3. Creating proper input/label pairs
    
    Args:
        texts: List of input texts
        tokenizer: Configured tokenizer
        chunk_size: Size of each chunk in tokens
        overlap_size: Overlap between consecutive chunks
        min_chunk_size: Minimum chunk size to keep
        
    Returns:
        List of tokenized chunks
    """
    
    chunks = []
    
    for text in texts:
        # Tokenize full document
        tokens = tokenizer(
            text,
            return_tensors="np",
            add_special_tokens=True,
            truncation=False,  # Don't truncate, we'll chunk manually
        )
        
        input_ids = tokens['input_ids'][0]  # Remove batch dimension
        total_length = len(input_ids)
        
        if total_length < min_chunk_size:
            # Skip very short documents
            continue
        
        # Create overlapping chunks
        start_pos = 0
        while start_pos < total_length:
            end_pos = min(start_pos + chunk_size, total_length)
            
            # Extract chunk
            chunk_tokens = input_ids[start_pos:end_pos]
            
            # Skip if chunk is too small
            if len(chunk_tokens) < min_chunk_size:
                break
            
            # Create input/label pair
            # For causal LM: labels are input_ids shifted by 1
            chunk_input = chunk_tokens[:-1] if len(chunk_tokens) > 1 else chunk_tokens
            chunk_labels = chunk_tokens[1:] if len(chunk_tokens) > 1 else chunk_tokens
            
            # Pad to chunk_size if needed
            if len(chunk_input) < chunk_size - 1:
                pad_length = (chunk_size - 1) - len(chunk_input)
                chunk_input = jnp.concatenate([
                    chunk_input,
                    jnp.full(pad_length, tokenizer.pad_token_id)
                ])
                chunk_labels = jnp.concatenate([
                    chunk_labels,
                    jnp.full(pad_length, -100)  # Ignore padding in loss
                ])
            
            chunks.append({
                'input_ids': chunk_input.astype(jnp.int32),
                'labels': chunk_labels.astype(jnp.int32),
                'attention_mask': (chunk_input != tokenizer.pad_token_id).astype(jnp.int32),
            })
            
            # Move to next chunk with overlap
            if end_pos >= total_length:
                break
            
            start_pos = end_pos - overlap_size
    
    logger.info(f"Created {len(chunks)} chunks from {len(texts)} documents")
    
    return chunks


def batch_tokenize(
    texts: List[str],
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 1000,
    max_length: int = 8192,
    **kwargs
) -> List[Dict[str, jnp.ndarray]]:
    """
    Tokenize texts in batches for memory efficiency.
    
    Args:
        texts: List of input texts
        tokenizer: Configured tokenizer
        batch_size: Number of texts to process at once
        max_length: Maximum sequence length
        
    Returns:
        List of tokenized batches
    """
    
    batches = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        batch_tokens = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="np",
            return_attention_mask=True,
            add_special_tokens=True,
        )
        
        # Convert to JAX arrays
        batch_data = {
            'input_ids': jnp.array(batch_tokens['input_ids'], dtype=jnp.int32),
            'attention_mask': jnp.array(batch_tokens['attention_mask'], dtype=jnp.int32),
        }
        
        # Create labels (shifted input_ids)
        batch_data['labels'] = jnp.concatenate([
            batch_data['input_ids'][:, 1:],
            jnp.full((batch_data['input_ids'].shape[0], 1), -100)
        ], axis=1)
        
        batches.append(batch_data)
    
    return batches


def decode_tokens(
    token_ids: Union[jnp.ndarray, List[int]],
    tokenizer: PreTrainedTokenizerFast,
    skip_special_tokens: bool = True,
) -> str:
    """
    Decode token IDs back to text.
    
    Args:
        token_ids: Token IDs to decode
        tokenizer: Tokenizer to use for decoding
        skip_special_tokens: Whether to skip special tokens
        
    Returns:
        Decoded text
    """
    
    # Convert JAX array to list if needed
    if isinstance(token_ids, jnp.ndarray):
        token_ids = token_ids.tolist()
    
    # Remove padding tokens
    if tokenizer.pad_token_id is not None:
        token_ids = [t for t in token_ids if t != tokenizer.pad_token_id]
    
    # Decode
    text = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    return text


def get_tokenizer_stats(tokenizer: PreTrainedTokenizerFast) -> Dict[str, Any]:
    """
    Get statistics about tokenizer configuration.
    
    Args:
        tokenizer: Tokenizer to analyze
        
    Returns:
        Dictionary with tokenizer statistics
    """
    
    stats = {
        'vocab_size': len(tokenizer),
        'model_max_length': tokenizer.model_max_length,
        'padding_side': tokenizer.padding_side,
        'truncation_side': tokenizer.truncation_side,
        'special_tokens': {
            'pad_token': tokenizer.pad_token,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token': tokenizer.eos_token,
            'eos_token_id': tokenizer.eos_token_id,
            'bos_token': tokenizer.bos_token,
            'bos_token_id': tokenizer.bos_token_id,
            'unk_token': tokenizer.unk_token,
            'unk_token_id': tokenizer.unk_token_id,
        },
        'added_tokens': len(tokenizer.added_tokens_encoder),
    }
    
    return stats


def validate_tokenizer(tokenizer: PreTrainedTokenizerFast) -> bool:
    """
    Validate tokenizer configuration.
    
    Args:
        tokenizer: Tokenizer to validate
        
    Returns:
        True if validation passes
    """
    
    logger.info("Validating tokenizer configuration...")
    
    # Check required tokens
    if tokenizer.pad_token_id is None:
        logger.error("Pad token not configured")
        return False
    
    if tokenizer.eos_token_id is None:
        logger.error("EOS token not configured")
        return False
    
    # Test tokenization
    test_text = "Hello, world! This is a test."
    try:
        tokens = tokenizer(test_text, return_tensors="np")
        decoded = tokenizer.decode(tokens['input_ids'][0])
        logger.info(f"Test tokenization successful: '{test_text}' -> {len(tokens['input_ids'][0])} tokens")
    except Exception as e:
        logger.error(f"Tokenization test failed: {e}")
        return False
    
    logger.info("✓ Tokenizer validation passed")
    return True


# Preset configurations
def get_gpt2_tokenizer_config() -> TokenizerConfig:
    """Get standard GPT-2 tokenizer configuration."""
    return TokenizerConfig(
        tokenizer_name="gpt2",
        vocab_size=50257,
        max_length=32768,
        pad_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        bos_token=None,
        unk_token="<|endoftext|>",
    )


def get_fineweb_tokenizer_config() -> TokenizerConfig:
    """Get tokenizer configuration optimized for FineWeb dataset."""
    return TokenizerConfig(
        tokenizer_name="gpt2",
        vocab_size=50257,
        max_length=65536,  # Longer sequences for FineWeb
        pad_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        bos_token=None,
        unk_token="<|endoftext|>",
        padding_side="right",
        truncation_side="right",
    )