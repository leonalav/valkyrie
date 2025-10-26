"""HRM Integration for Valkyrie Model

Integrates the Hierarchical Reasoning Model (HRM) with the Valkyrie architecture.
Provides plan token generation and hierarchical reasoning capabilities.

Based on the HRM paper and mathematical specifications in valkyrie_math_part.md
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import math
from typing import Optional, Tuple, Dict, NamedTuple
from functools import partial

from .hrm.models.hrm_inner import HRMInner, HRMInnerCarry
from .modules import RMSNorm


class HRMPlannerState(NamedTuple):
    """State for HRM planner during generation."""
    z_H: jnp.ndarray  # High-level reasoning state
    z_L: jnp.ndarray  # Low-level reasoning state
    plan_tokens: jnp.ndarray  # Generated plan tokens
    halt_probs: jnp.ndarray  # ACT halting probabilities


class ValkyrieHRMPlanner(nn.Module):
    """HRM Planner module for generating plan tokens.
    
    Generates plan tokens that serve as global tokens for BigBird attention.
    These tokens encode high-level reasoning and planning information.
    """
    config: 'ValkyrieConfig'
    
    def setup(self):
        # Plan token writer - projects HRM states to plan tokens
        self.plan_token_writer = nn.Dense(
            self.config.d_model,
            use_bias=self.config.use_bias,
            name="plan_token_writer"
        )
        
        # Normalization for plan tokens
        self.plan_norm = RMSNorm(
            self.config.d_model, 
            eps=self.config.layer_norm_eps
        )
        
        # HRM inner reasoning module (adapted configuration)
        self.hrm_inner = HRMInner(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.d_model,
            seq_len=self.config.hrm_plan_length,
            H_cycles=self.config.hrm_H_cycles,
            L_cycles=self.config.hrm_L_cycles,
            H_layers=self.config.hrm_H_layers,
            L_layers=self.config.hrm_L_layers,
            num_heads=self.config.n_heads,
            num_key_value_heads=self.config.n_kv_heads,
            intermediate_size=self.config.hrm_intermediate_size,
            eps=self.config.layer_norm_eps,
            pos_encodings="rope",
            rope_theta=self.config.rope_theta,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32
        )
    
    def generate_plan_tokens(self,
                           input_embeddings: jnp.ndarray,
                           hrm_state: Optional[HRMPlannerState] = None,
                           training: bool = False) -> Tuple[jnp.ndarray, HRMPlannerState]:
        """Generate plan tokens from input embeddings.
        
        Args:
            input_embeddings: Input token embeddings [batch, seq_len, d_model]
            hrm_state: Previous HRM state for generation
            training: Whether in training mode
            
        Returns:
            Tuple of (plan_tokens, new_hrm_state)
        """
        batch_size, seq_len, d_model = input_embeddings.shape
        
        # Initialize HRM state if not provided
        if hrm_state is None:
            # Initialize with input summary
            input_summary = jnp.mean(input_embeddings, axis=1, keepdims=True)  # [batch, 1, d_model]
            input_summary = jnp.tile(input_summary, (1, self.config.hrm_plan_length, 1))
            
            z_H = input_summary
            z_L = input_summary
            plan_tokens = jnp.zeros((batch_size, self.config.bigbird_num_global_tokens, d_model))
            halt_probs = jnp.zeros((batch_size, self.config.hrm_plan_length, 2))
        else:
            z_H = hrm_state.z_H
            z_L = hrm_state.z_L
            plan_tokens = hrm_state.plan_tokens
            halt_probs = hrm_state.halt_probs
        
        # Run HRM reasoning (simplified for integration)
        # In full implementation, this would run the complete HRM cycles
        carry = HRMInnerCarry(z_H=z_H, z_L=z_L)
        
        # One-step gradient reasoning
        if training:
            # Detach carry for one-step gradient
            detached_carry = HRMInnerCarry(
                z_H=jax.lax.stop_gradient(carry.z_H),
                z_L=jax.lax.stop_gradient(carry.z_L)
            )
            
            # Perform one reasoning step with gradients
            new_z_H = self.hrm_inner.H_level(
                detached_carry.z_H,
                input_injection=jnp.mean(input_embeddings, axis=1, keepdims=True)
            )
            new_z_L = self.hrm_inner.L_level(
                detached_carry.z_L,
                input_injection=new_z_H
            )
        else:
            # During inference, run full reasoning
            new_z_H = carry.z_H
            new_z_L = carry.z_L
            
            # Simplified reasoning step
            input_injection = jnp.mean(input_embeddings, axis=1, keepdims=True)
            new_z_H = self.hrm_inner.H_level(new_z_H, input_injection=input_injection)
            new_z_L = self.hrm_inner.L_level(new_z_L, input_injection=new_z_H)
        
        # Generate plan tokens from HRM states
        # Combine H and L level information
        combined_state = new_z_H + new_z_L  # [batch, plan_length, d_model]
        
        # Project to plan tokens
        raw_plan_tokens = self.plan_token_writer(combined_state)
        plan_tokens = self.plan_norm(raw_plan_tokens)
        
        # Truncate or pad to match global token count
        if plan_tokens.shape[1] > self.config.bigbird_num_global_tokens:
            plan_tokens = plan_tokens[:, :self.config.bigbird_num_global_tokens]
        elif plan_tokens.shape[1] < self.config.bigbird_num_global_tokens:
            pad_size = self.config.bigbird_num_global_tokens - plan_tokens.shape[1]
            padding = jnp.zeros((batch_size, pad_size, d_model))
            plan_tokens = jnp.concatenate([plan_tokens, padding], axis=1)
        
        # Compute ACT halting probabilities using HRM inner's q_head
        # The q_head expects the first position of the H-level state
        q_logits = self.hrm_inner.q_head(new_z_H[:, 0])  # [batch, 2]
        halt_probs = jax.nn.softmax(q_logits, axis=-1)  # [batch, 2]
        
        # Expand halt_probs to match plan_length for consistency
        halt_probs_expanded = jnp.tile(halt_probs[:, None, :], (1, self.config.hrm_plan_length, 1))
        
        # Create new HRM state
        new_hrm_state = HRMPlannerState(
            z_H=new_z_H,
            z_L=new_z_L,
            plan_tokens=plan_tokens,
            halt_probs=halt_probs_expanded
        )
        
        return plan_tokens, new_hrm_state
    
    def __call__(self,
                 input_embeddings: jnp.ndarray,
                 hrm_state: Optional[HRMPlannerState] = None,
                 training: bool = False) -> Tuple[jnp.ndarray, HRMPlannerState]:
        """Forward pass of HRM planner."""
        return self.generate_plan_tokens(input_embeddings, hrm_state, training)


class ValkyrieHRMExecutor(nn.Module):
    """HRM Executor module for processing plan tokens.
    
    Processes the plan tokens generated by the planner and integrates
    them with the main sequence processing.
    """
    config: 'ValkyrieConfig'
    
    def setup(self):
        # Plan token reader - processes plan tokens for execution
        self.plan_token_reader = nn.Dense(
            self.config.d_model,
            use_bias=self.config.use_bias,
            name="plan_token_reader"
        )
        
        # Gated mixer for combining plan and sequence information
        self.gate_proj = nn.Dense(
            self.config.d_model,
            use_bias=self.config.use_bias,
            name="gate_proj"
        )
        
        self.mix_proj = nn.Dense(
            self.config.d_model,
            use_bias=self.config.use_bias,
            name="mix_proj"
        )
        
        # Normalization
        self.executor_norm = RMSNorm(
            self.config.d_model,
            eps=self.config.layer_norm_eps
        )
    
    def __call__(self,
                 sequence_states: jnp.ndarray,
                 plan_tokens: jnp.ndarray,
                 training: bool = False) -> jnp.ndarray:
        """Execute plan-guided processing.
        
        Args:
            sequence_states: Main sequence hidden states [batch, seq_len, d_model]
            plan_tokens: Plan tokens from planner [batch, num_global_tokens, d_model]
            training: Whether in training mode
            
        Returns:
            Enhanced sequence states with plan guidance
        """
        batch_size, seq_len, d_model = sequence_states.shape
        
        # Process plan tokens
        processed_plans = self.plan_token_reader(plan_tokens)
        
        # Aggregate plan information (mean pooling)
        plan_summary = jnp.mean(processed_plans, axis=1, keepdims=True)  # [batch, 1, d_model]
        plan_summary = jnp.tile(plan_summary, (1, seq_len, 1))  # [batch, seq_len, d_model]
        
        # Gated mixing of sequence and plan information
        gate = jax.nn.sigmoid(self.gate_proj(sequence_states))
        mix = self.mix_proj(plan_summary)
        
        # Combine with gating
        enhanced_states = sequence_states + gate * mix
        
        # Normalize output
        return self.executor_norm(enhanced_states)


class ValkyrieHRM(nn.Module):
    """Complete HRM integration for Valkyrie.
    
    Combines planner and executor for full hierarchical reasoning.
    """
    config: 'ValkyrieConfig'
    
    def setup(self):
        self.planner = ValkyrieHRMPlanner(self.config)
        self.executor = ValkyrieHRMExecutor(self.config)
    
    def __call__(self,
                 input_embeddings: jnp.ndarray,
                 sequence_states: jnp.ndarray,
                 hrm_state: Optional[HRMPlannerState] = None,
                 training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, HRMPlannerState]:
        """Full HRM forward pass.
        
        Args:
            input_embeddings: Input token embeddings [batch, seq_len, d_model]
            sequence_states: Current sequence hidden states [batch, seq_len, d_model]
            hrm_state: Previous HRM state
            training: Whether in training mode
            
        Returns:
            Tuple of (plan_tokens, enhanced_sequence_states, new_hrm_state)
        """
        # Generate plan tokens
        plan_tokens, new_hrm_state = self.planner(
            input_embeddings, hrm_state, training
        )
        
        # Execute plan-guided processing
        enhanced_states = self.executor(
            sequence_states, plan_tokens, training
        )
        
        return plan_tokens, enhanced_states, new_hrm_state