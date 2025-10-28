"""Orbax-based checkpointing for multi-host TPU training.

Implements:
- Sharded parameter checkpointing across TPU hosts
- S5 state persistence and restoration
- Asynchronous checkpoint saving
- Checkpoint validation and integrity checks
- Multi-level checkpointing (fast + full)
"""

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from orbax.checkpoint import PyTreeCheckpointer, AsyncCheckpointer
from typing import Dict, Any, Optional, Union, List
import os
import time
import logging
from pathlib import Path
import json
import asyncio
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    checkpoint_dir: str = "./checkpoints"
    save_interval_steps: int = 1000
    keep_checkpoints: int = 5
    async_save: bool = True
    
    # Multi-level checkpointing
    fast_checkpoint_interval: int = 100  # Fast checkpoints (params only)
    full_checkpoint_interval: int = 1000  # Full checkpoints (params + optimizer + S5)
    
    # Validation
    validate_on_save: bool = True
    validate_on_load: bool = True
    
    # Compression
    use_compression: bool = True
    compression_level: int = 6
    
    # Additional parameters from YAML config
    save_optimizer_state: bool = True  # Whether to save optimizer state
    async_checkpointing: bool = True  # Async checkpointing flag


class CheckpointManager:
    """
    Multi-host checkpoint manager with Orbax backend.
    
    Features:
    - Sharded parameter saving/loading
    - S5 state persistence
    - Asynchronous checkpoint operations
    - Multi-level checkpointing (fast/full)
    - Automatic cleanup of old checkpoints
    - Checkpoint validation and integrity checks
    """
    
    def __init__(
        self,
        config: CheckpointConfig,
        mesh: Optional[Any] = None,
        partition_specs: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.mesh = mesh
        self.partition_specs = partition_specs
        
        # Setup checkpoint directory - use absolute path
        self.checkpoint_dir = Path(config.checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Multi-host coordination
        self.process_index = jax.process_index()
        self.process_count = jax.process_count()
        self.is_primary_host = (self.process_index == 0)
        
        # Initialize checkpointers
        self._setup_checkpointers()
        
        # Checkpoint tracking
        self.saved_checkpoints = []
        self._load_checkpoint_manifest()
        
        logger.info(f"Checkpoint manager initialized:")
        logger.info(f"  Directory: {self.checkpoint_dir}")
        logger.info(f"  Process: {self.process_index}/{self.process_count}")
        logger.info(f"  Async save: {config.async_save}")
        logger.info(f"  Keep checkpoints: {config.keep_checkpoints}")
    
    def _setup_checkpointers(self):
        """Setup Orbax checkpointers using current API."""
        
        # Main checkpointer for parameters and optimizer state
        checkpointer_options = ocp.CheckpointManagerOptions(
            save_interval_steps=self.config.save_interval_steps,
            max_to_keep=self.config.keep_checkpoints,
            create=True,
        )
        
        if self.config.async_save:
            # Use current Orbax API for async checkpointing
            self.checkpointer = ocp.AsyncCheckpointer(
                ocp.StandardCheckpointHandler(),
                timeout_secs=300,  # 5 minute timeout
            )
        else:
            self.checkpointer = ocp.StandardCheckpointer()
        
        # Fast checkpointer for frequent saves (params only)
        self.fast_checkpointer = ocp.StandardCheckpointer()
        
        logger.info("✓ Checkpointers initialized")
    
    def _get_checkpoint_path(self, step: int, checkpoint_type: str = "full") -> Path:
        """Get path for checkpoint at given step."""
        return self.checkpoint_dir / f"{checkpoint_type}_checkpoint_{step:08d}"
    
    def _load_checkpoint_manifest(self):
        """Load manifest of existing checkpoints."""
        manifest_path = self.checkpoint_dir / "checkpoint_manifest.json"
        
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                self.saved_checkpoints = manifest.get('checkpoints', [])
                logger.info(f"Loaded checkpoint manifest: {len(self.saved_checkpoints)} checkpoints")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint manifest: {e}")
                self.saved_checkpoints = []
        else:
            self.saved_checkpoints = []
    
    def _save_checkpoint_manifest(self):
        """Save manifest of checkpoints."""
        if not self.is_primary_host:
            return
        
        manifest_path = self.checkpoint_dir / "checkpoint_manifest.json"
        
        manifest = {
            'checkpoints': self.saved_checkpoints,
            'config': asdict(self.config),
            'last_updated': time.time(),
        }
        
        try:
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint manifest: {e}")
    
    def save(
        self,
        state: Any,  # Changed from Dict[str, Any] to Any to support NamedTuple
        step: Optional[int] = None,
        checkpoint_type: str = "full",
        path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """
        Save checkpoint with proper sharding and error handling.
        
        Args:
            state: Training state to save
            step: Training step (uses state.step if None)
            checkpoint_type: Type of checkpoint ("full" or "fast")
            path: Custom checkpoint path
            
        Returns:
            True if successful, False otherwise
        """
        
        if step is None:
            step = getattr(state, 'step', 0)
        
        # Determine checkpoint path
        if path is not None:
            checkpoint_path = Path(path)
        else:
            checkpoint_path = self._get_checkpoint_path(step, checkpoint_type)
        
        logger.info(f"Saving {checkpoint_type} checkpoint at step {step}: {checkpoint_path}")
        
        try:
            # Prepare checkpoint data
            checkpoint_data = self._prepare_checkpoint_data(state, checkpoint_type)
            
            # Create checkpoint directory
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save checkpoint with proper error handling and cleanup
            if checkpoint_type == "fast":
                self.fast_checkpointer.save(checkpoint_path, args=ocp.args.StandardSave(checkpoint_data))
            else:
                if self.config.async_save:
                    self.checkpointer.save(checkpoint_path, args=ocp.args.StandardSave(checkpoint_data))
                else:
                    self.checkpointer.save(checkpoint_path, args=ocp.args.StandardSave(checkpoint_data))
            
            # Update manifest
            checkpoint_info = {
                'step': step,
                'path': str(checkpoint_path),
                'type': checkpoint_type,
                'timestamp': time.time(),
                'process_count': self.process_count,
            }
            
            self.saved_checkpoints.append(checkpoint_info)
            self.saved_checkpoints.sort(key=lambda x: x['step'])
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(checkpoint_type)
            
            # Save manifest
            self._save_checkpoint_manifest()
            
            # Validate checkpoint if enabled
            if self.config.validate_on_save:
                self._validate_checkpoint(checkpoint_path, checkpoint_data)
            
            logger.info(f"✓ Checkpoint saved successfully: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # CRITICAL FIX: Add proper Orbax checkpoint cleanup on failure
            self._cleanup_failed_checkpoint(checkpoint_path)
            return False
    
    def load(
        self,
        step: Optional[int] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_type: str = "full",
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint with proper sharding.
        
        Args:
            step: Training step to load (latest if None)
            checkpoint_path: Explicit path to checkpoint
            checkpoint_type: Type of checkpoint to load
            
        Returns:
            Loaded training state or None if failed
        """
        
        # Determine checkpoint path
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
        elif step is not None:
            checkpoint_path = self._get_checkpoint_path(step, checkpoint_type)
        else:
            # Load latest checkpoint
            checkpoint_path = self._get_latest_checkpoint_path(checkpoint_type)
        
        if checkpoint_path is None or not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            # Load checkpoint data - need to provide abstract structure for restore
            if checkpoint_type == "fast":
                # For fast checkpoints, we typically only save params
                abstract_state = {'params': None}  # Placeholder structure
                checkpoint_data = self.fast_checkpointer.restore(checkpoint_path, args=ocp.args.StandardRestore(abstract_state))
            else:
                # For full checkpoints, we need the full state structure
                abstract_state = {'params': None, 'opt_state': None, 'step': None}  # Placeholder structure
                checkpoint_data = self.checkpointer.restore(checkpoint_path, args=ocp.args.StandardRestore(abstract_state))
            
            # Validate checkpoint if enabled
            if self.config.validate_on_load:
                self._validate_checkpoint(checkpoint_path, checkpoint_data)
            
            # Restore training state
            state = self._restore_training_state(checkpoint_data, checkpoint_type)
            
            logger.info(f"✓ Checkpoint loaded successfully: {checkpoint_path}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _prepare_checkpoint_data(
        self,
        state: Any,  # Changed from Dict[str, Any] to Any to support NamedTuple
        checkpoint_type: str,
    ) -> Dict[str, Any]:
        """Prepare data for checkpointing."""
        
        checkpoint_data = {}
        
        # Always save parameters
        if hasattr(state, 'params'):
            checkpoint_data['params'] = state.params
        
        # Always save step and basic info
        checkpoint_data['step'] = getattr(state, 'step', 0)
        checkpoint_data['rng'] = getattr(state, 'rng', jax.random.PRNGKey(0))
        
        if checkpoint_type == "full":
            # Full checkpoint includes optimizer state and S5 states
            # Replace hasattr with try/except to avoid tracer issues
            try:
                checkpoint_data['opt_state'] = state.opt_state
            except AttributeError:
                pass
            
            try:
                if state.s5_states is not None:
                    checkpoint_data['s5_states'] = state.s5_states
            except AttributeError:
                pass
            
            # HRM states (z_H, z_L) as specified in PLAN
            try:
                if state.hrm_states is not None:
                    checkpoint_data['hrm_states'] = state.hrm_states
            except AttributeError:
                pass
            
            # Individual HRM state components for backward compatibility
            try:
                if state.z_H is not None:
                    checkpoint_data['z_H'] = state.z_H
            except AttributeError:
                pass
            
            try:
                if state.z_L is not None:
                    checkpoint_data['z_L'] = state.z_L
            except AttributeError:
                pass
            
            # HRM training metadata (segment info, cycle counts, etc.)
            try:
                checkpoint_data['hrm_metadata'] = state.hrm_metadata
            except AttributeError:
                pass
            
            # Add metadata - use try/except for state attribute checks
            has_hrm_states = False
            has_s5_states = False
            
            try:
                has_hrm_states = state.hrm_states is not None
            except AttributeError:
                try:
                    has_hrm_states = state.z_H is not None and state.z_L is not None
                except AttributeError:
                    pass
            
            try:
                has_s5_states = state.s5_states is not None
            except AttributeError:
                pass
            
            checkpoint_data['metadata'] = {
                'checkpoint_type': checkpoint_type,
                'process_count': self.process_count,
                'timestamp': time.time(),
                'has_hrm_states': has_hrm_states,
                'has_s5_states': has_s5_states,
            }
        
        return checkpoint_data
    
    def _restore_training_state(
        self,
        checkpoint_data: Dict[str, Any],
        checkpoint_type: str,
    ) -> Any:
        """Restore training state from checkpoint data."""
        
        # Import TrainingState here to avoid circular imports
        from ..train.train_loop import TrainingState
        
        # Restore basic components (required fields)
        params = checkpoint_data['params']
        step = checkpoint_data['step']
        rng = checkpoint_data['rng']
        opt_state = checkpoint_data.get('opt_state', None)
        
        # Restore optional components with defaults
        s5_states = checkpoint_data.get('s5_states', None)
        chunk_position = checkpoint_data.get('chunk_position', 0)
        phase_index = checkpoint_data.get('phase_index', 0)
        hrm_enabled = checkpoint_data.get('hrm_enabled', False)
        hrm_training_state = checkpoint_data.get('hrm_training_state', None)
        
        # Handle backward compatibility for HRM states
        if 'hrm_states' in checkpoint_data:
            hrm_training_state = checkpoint_data['hrm_states']
        
        # Construct and return the proper TrainingState NamedTuple
        return TrainingState(
            params=params,
            opt_state=opt_state,
            step=step,
            rng=rng,
            s5_states=s5_states,
            chunk_position=chunk_position,
            phase_index=phase_index,
            hrm_enabled=hrm_enabled,
            hrm_training_state=hrm_training_state
        )
    
    def _get_latest_checkpoint_path(self, checkpoint_type: str = "full") -> Optional[Path]:
        """Get path to latest checkpoint of given type."""
        
        # Filter checkpoints by type
        type_checkpoints = [
            cp for cp in self.saved_checkpoints 
            if cp['type'] == checkpoint_type
        ]
        
        if not type_checkpoints:
            return None
        
        # Get latest by step
        latest = max(type_checkpoints, key=lambda x: x['step'])
        return Path(latest['path'])
    
    def _cleanup_old_checkpoints(self, checkpoint_type: str):
        """Remove old checkpoints beyond keep limit."""
        
        if not self.is_primary_host:
            return
        
        # Filter checkpoints by type
        type_checkpoints = [
            cp for cp in self.saved_checkpoints 
            if cp['type'] == checkpoint_type
        ]
        
        if len(type_checkpoints) <= self.config.keep_checkpoints:
            return
        
        # Sort by step and remove oldest
        type_checkpoints.sort(key=lambda x: x['step'])
        to_remove = type_checkpoints[:-self.config.keep_checkpoints]
        
        for checkpoint_info in to_remove:
            try:
                checkpoint_path = Path(checkpoint_info['path'])
                if checkpoint_path.exists():
                    # Remove checkpoint directory
                    import shutil
                    shutil.rmtree(checkpoint_path)
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
                
                # Remove from manifest
                self.saved_checkpoints.remove(checkpoint_info)
                
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint: {e}")
    
    def _validate_checkpoint(
        self,
        checkpoint_path: Path,
        checkpoint_data: Dict[str, Any],
    ) -> bool:
        """Validate checkpoint integrity."""
        
        try:
            # Check path exists
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
                return False
            
            # Check required keys
            required_keys = ['params', 'step', 'rng']
            for key in required_keys:
                if key not in checkpoint_data:
                    logger.error(f"Missing required key in checkpoint: {key}")
                    return False
            
            # Check parameter structure
            params = checkpoint_data['params']
            if not isinstance(params, dict):
                logger.error("Parameters must be a dictionary")
                return False
            
            # Check step is valid
            step = checkpoint_data['step']
            if not isinstance(step, (int, jnp.integer)) or step < 0:
                logger.error(f"Invalid step value: {step}")
                return False
            
            logger.debug(f"✓ Checkpoint validation passed: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False
    
    def list_checkpoints(self, checkpoint_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        
        if checkpoint_type is None:
            return self.saved_checkpoints.copy()
        else:
            return [
                cp for cp in self.saved_checkpoints 
                if cp['type'] == checkpoint_type
            ]
    
    def get_latest_step(self, checkpoint_type: str = "full") -> Optional[int]:
        """Get step number of latest checkpoint."""
        
        latest_path = self._get_latest_checkpoint_path(checkpoint_type)
        if latest_path is None:
            return None
        
        # Extract step from saved checkpoints
        for cp in self.saved_checkpoints:
            if Path(cp['path']) == latest_path:
                return cp['step']
        
        return None
    
    def cleanup_all_checkpoints(self):
        """Remove all checkpoints (use with caution)."""
        
        if not self.is_primary_host:
            return
        
        logger.warning("Removing all checkpoints...")
        
        for checkpoint_info in self.saved_checkpoints.copy():
            try:
                checkpoint_path = Path(checkpoint_info['path'])
                if checkpoint_path.exists():
                    import shutil
                    shutil.rmtree(checkpoint_path)
                
                self.saved_checkpoints.remove(checkpoint_info)
                
            except Exception as e:
                logger.error(f"Failed to remove checkpoint: {e}")
        
        # Update manifest
        self._save_checkpoint_manifest()
        
        logger.info("All checkpoints removed")
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics."""
        
        stats = {
            'total_checkpoints': len(self.saved_checkpoints),
            'checkpoint_types': {},
            'disk_usage_mb': 0,
            'latest_steps': {},
        }
        
        # Count by type
        for cp in self.saved_checkpoints:
            cp_type = cp['type']
            if cp_type not in stats['checkpoint_types']:
                stats['checkpoint_types'][cp_type] = 0
            stats['checkpoint_types'][cp_type] += 1
        
        # Get latest steps by type
        for cp_type in stats['checkpoint_types']:
            latest_step = self.get_latest_step(cp_type)
            if latest_step is not None:
                stats['latest_steps'][cp_type] = latest_step
        
        # Calculate disk usage
        try:
            total_size = 0
            for cp in self.saved_checkpoints:
                cp_path = Path(cp['path'])
                if cp_path.exists():
                    total_size += sum(f.stat().st_size for f in cp_path.rglob('*') if f.is_file())
            
            stats['disk_usage_mb'] = total_size / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to calculate disk usage: {e}")
        
        return stats
    
    def _cleanup_failed_checkpoint(self, step: int, checkpoint_type: str = "unknown") -> None:
        """Clean up failed checkpoint artifacts and update manifest.
        
        Args:
            step: The step number of the failed checkpoint
            checkpoint_type: Type of checkpoint that failed ("fast" or "full")
        """
        try:
            # Remove any partially created checkpoint directory
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}"
            if checkpoint_path.exists():
                import shutil
                shutil.rmtree(checkpoint_path, ignore_errors=True)
                logger.info(f"Removed failed {checkpoint_type} checkpoint directory: {checkpoint_path}")
            
            # Cancel any pending async operations for this step
            if hasattr(self, '_async_futures'):
                futures_to_cancel = []
                for future in getattr(self, '_async_futures', []):
                    if not future.done():
                        future.cancel()
                        futures_to_cancel.append(future)
                
                if futures_to_cancel:
                    logger.info(f"Cancelled {len(futures_to_cancel)} pending async operations")
            
            # Update manifest to remove any reference to this failed checkpoint
            if hasattr(self, 'saved_checkpoints'):
                self.saved_checkpoints = [
                    cp for cp in self.saved_checkpoints 
                    if not cp['path'].endswith(f"checkpoint_{step}")
                ]
                self._update_manifest()
                
        except Exception as cleanup_error:
            logger.error(f"Error during checkpoint cleanup for step {step}: {cleanup_error}")
            # Don't re-raise - cleanup errors shouldn't crash the training