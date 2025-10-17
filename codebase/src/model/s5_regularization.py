"""
S5 Regularization utilities for handling complex outputs and encouraging conjugate symmetry.

This module provides regularization functions to encourage the S5 model to learn
proper conjugate symmetry, which should naturally drive imaginary components to zero
while preserving gradient flow during training.
"""

import jax
import jax.numpy as jnp
from typing import Union, Dict, Any


def imaginary_part_regularization(
    outputs: jnp.ndarray,
    regularization_weight: float = 1e-3,
    reduction: str = 'mean'
) -> jnp.ndarray:
    """
    Compute regularization loss to encourage small imaginary parts in S5 outputs.
    
    This regularization encourages the model to learn conjugate symmetry naturally,
    which should theoretically produce real outputs. During training, we preserve
    gradient flow through complex outputs while penalizing large imaginary components.
    
    Args:
        outputs: Complex or real outputs from S5 layer [batch, seq_len, d_model]
        regularization_weight: Weight for the imaginary part penalty
        reduction: How to reduce the loss ('mean', 'sum', 'none')
        
    Returns:
        Regularization loss scalar
    """
    # Handle both complex and real inputs
    if jnp.iscomplexobj(outputs):
        # Extract imaginary part and compute squared magnitude
        imag_part = jnp.imag(outputs)
        imag_loss = jnp.sum(imag_part ** 2, axis=-1)  # Sum over d_model dimension
        
        if reduction == 'mean':
            imag_loss = jnp.mean(imag_loss)
        elif reduction == 'sum':
            imag_loss = jnp.sum(imag_loss)
        elif reduction == 'none':
            pass  # Keep per-sample losses
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
            
        return regularization_weight * imag_loss
    else:
        # If outputs are already real, no regularization needed
        return jnp.array(0.0)


def conjugate_symmetry_loss(
    Lambda: jnp.ndarray,
    B_tilde: jnp.ndarray,
    C_tilde: jnp.ndarray,
    symmetry_weight: float = 1e-4
) -> jnp.ndarray:
    """
    Compute loss to encourage proper conjugate symmetry in S5 parameters.
    
    This loss directly encourages the parameters to maintain conjugate symmetry,
    which is the mathematical foundation for real outputs in S5 models.
    
    Args:
        Lambda: Complex eigenvalues [state_dim]
        B_tilde: Complex input matrix [state_dim, d_model]
        C_tilde: Complex output matrix [d_model, state_dim]
        symmetry_weight: Weight for the conjugate symmetry penalty
        
    Returns:
        Conjugate symmetry loss scalar
    """
    # Assume parameters are constructed as conjugate pairs
    # Lambda: [λ₁, λ₁*, λ₂, λ₂*, ...] where * denotes conjugate
    # Check that consecutive pairs are conjugates
    
    state_dim = Lambda.shape[0]
    if state_dim % 2 != 0:
        raise ValueError("State dimension must be even for conjugate pairs")
    
    # Split into pairs and check conjugate symmetry
    Lambda_pairs = Lambda.reshape(-1, 2)  # [state_dim//2, 2]
    B_pairs = B_tilde.reshape(-1, 2, B_tilde.shape[-1])  # [state_dim//2, 2, d_model]
    C_pairs = C_tilde.reshape(C_tilde.shape[0], -1, 2)  # [d_model, state_dim//2, 2]
    
    # Compute conjugate symmetry violations
    lambda_violation = jnp.mean((Lambda_pairs[:, 0] - jnp.conj(Lambda_pairs[:, 1])) ** 2)
    b_violation = jnp.mean((B_pairs[:, 0, :] - jnp.conj(B_pairs[:, 1, :])) ** 2)
    c_violation = jnp.mean((C_pairs[:, :, 0] - jnp.conj(C_pairs[:, :, 1])) ** 2)
    
    total_violation = lambda_violation + b_violation + c_violation
    
    return symmetry_weight * jnp.real(total_violation)


def s5_training_loss(
    outputs: jnp.ndarray,
    targets: jnp.ndarray,
    s5_params: Dict[str, jnp.ndarray],
    base_loss_fn: callable,
    imaginary_weight: float = 1e-3,
    symmetry_weight: float = 1e-4,
    **loss_kwargs
) -> Dict[str, jnp.ndarray]:
    """
    Compute total training loss for S5 model including regularization terms.
    
    Args:
        outputs: Model outputs (complex during training, real during inference)
        targets: Target labels
        s5_params: Dictionary containing S5 parameters (Lambda, B_tilde, C_tilde)
        base_loss_fn: Base loss function (e.g., cross-entropy)
        imaginary_weight: Weight for imaginary part regularization
        symmetry_weight: Weight for conjugate symmetry regularization
        **loss_kwargs: Additional arguments for base loss function
        
    Returns:
        Dictionary containing total loss and individual loss components
    """
    # Handle complex outputs during training
    if jnp.iscomplexobj(outputs):
        # Use real part for base loss computation
        real_outputs = jnp.real(outputs).astype(jnp.float32)
        base_loss = base_loss_fn(real_outputs, targets, **loss_kwargs)
        
        # Add imaginary part regularization
        imag_reg = imaginary_part_regularization(outputs, imaginary_weight)
        
        # Add conjugate symmetry regularization if parameters are provided
        if s5_params and all(key in s5_params for key in ['Lambda', 'B_tilde', 'C_tilde']):
            symmetry_reg = conjugate_symmetry_loss(
                s5_params['Lambda'],
                s5_params['B_tilde'], 
                s5_params['C_tilde'],
                symmetry_weight
            )
        else:
            symmetry_reg = jnp.array(0.0)
        
        total_loss = base_loss + imag_reg + symmetry_reg
        
        return {
            'total_loss': total_loss,
            'base_loss': base_loss,
            'imaginary_regularization': imag_reg,
            'symmetry_regularization': symmetry_reg
        }
    else:
        # Real outputs - just compute base loss
        base_loss = base_loss_fn(outputs, targets, **loss_kwargs)
        return {
            'total_loss': base_loss,
            'base_loss': base_loss,
            'imaginary_regularization': jnp.array(0.0),
            'symmetry_regularization': jnp.array(0.0)
        }