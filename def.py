# training
# requirements: pip install datasets transformers accelerate wandb
import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed
import time
import json
import warnings
from contextlib import nullcontext

#warnings are annoying so we'll skip this
warnings.filterwarnings("ignore", category=UserWarning, message=".*resume_download.*")

#configs for training
@dataclass
class TrainConfig:
    #dataset configuration
    dataset_name: str = "roneneldan/TinyStories"
    split: str = "train"
    val_split: str = "validation"
    val_size: float = 0.1
    max_length: int = 1024
    
    #model configuration
    model_config: Dict = field(default_factory=lambda: {
        "d_model": 768,
        "n_layers": 12, 
        "n_heads": 12,
        "max_position_embeddings": 1024,
        "ffn_dropout": 0.1,
        "attn_dropout": 0.0,
        "resid_dropout": 0.0
    })
    
    #training params
    per_device_batch_size: int = 4  # Reduced for stability
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size
    num_train_epochs: int = 3
    max_train_steps: Optional[int] = None
    
    # Learning rate and optimization  
    learning_rate: float = 3e-4  # Reduced from 5e-4 for stability
    lr_schedule: str = "cosine"
    weight_decay: float = 0.1  # Standard for transformers
    adam_betas: tuple = (0.9, 0.95)
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.03  # Reduced warmup
    max_grad_norm: float = 1.0
    
    # Checkpointing and evaluation
    save_every_n_steps: int = 2000
    eval_steps: int = 1000
    save_total_limit: int = 3
    early_stopping_patience: int = 10
    
    # Performance optimizations
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    compile_model: bool = False  # PyTorch 2.0 compilation
    
    # Data loading
    dataloader_drop_last: bool = True
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = False  # Can cause issues
    num_workers: int = 4
    
    # Logging and monitoring
    output_dir: str = "./valkyrie_tinystories_ckpts"
    logging_dir: str = "./logs"
    log_interval: int = 50
    report_to: Optional[str] = None  # "wandb", "tensorboard", or None
    
    # Training stability and recovery
    resume_from_checkpoint: Optional[str] = None
    save_optimizer_states: bool = True
    detect_anomaly: bool = False  # For debugging
    
    # Random seed
    seed: int = 42

# -------------------------
# Utility functions
# -------------------------
def get_model_size(model):
    """Calculate model size in millions of parameters"""
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_count / 1e6

def check_model_health(model):
    """Check for common model issues"""
    issues = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if torch.isnan(param).any():
            issues.append(f"NaN in parameter: {name}")
        if torch.isinf(param).any():
            issues.append(f"Inf in parameter: {name}")
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                issues.append(f"NaN in gradient: {name}")
            if torch.isinf(param.grad).any():
                issues.append(f"Inf in gradient: {name}")
    
    return issues

def estimate_memory_usage(model, batch_size, seq_len):
    """Estimate memory usage"""
    param_count = sum(p.numel() for p in model.parameters())
    # Rough estimation: parameters + gradients + optimizer states + activations
    param_mem = param_count * 4  # 4 bytes per float32
    grad_mem = param_mem  # Gradients
    optim_mem = param_mem * 2  # Adam has 2 states per parameter
    
    # Rough activation memory estimate
    hidden_size = getattr(model.config, 'd_model', 768)
    n_layers = getattr(model.config, 'n_layers', 12)
    activation_mem = batch_size * seq_len * hidden_size * n_layers * 4
    
    total_gb = (param_mem + grad_mem + optim_mem + activation_mem) / (1024**3)
    return total_gb

# -------------------------
# Training Setup
# -------------------------
def setup_training():
    cfg = TrainConfig()
    
    # Setup accelerator with better error handling
    accelerator = Accelerator(
        mixed_precision="fp16" if cfg.fp16 else ("bf16" if cfg.bf16 else "no"),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with=cfg.report_to,
        project_dir=cfg.logging_dir if cfg.report_to else None,
    )
    
    # Multi-GPU verification and diagnostics
    accelerator.print("=== GPU Setup Diagnostics ===")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Process index: {accelerator.process_index}")
    accelerator.print(f"Local process index: {accelerator.local_process_index}")
    accelerator.print(f"Device: {accelerator.device}")
    
    if torch.cuda.is_available():
        accelerator.print(f"CUDA devices available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            accelerator.print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    
    accelerator.print(f"Distributed training: {'Yes' if accelerator.num_processes > 1 else 'No'}")
    accelerator.print("=" * 30)
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Anomaly detection for debugging
    if cfg.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        accelerator.print("Anomaly detection enabled")
    
    return cfg, accelerator

def setup_tokenizer_and_data(cfg, accelerator):
    """Setup tokenizer and datasets with error handling"""
    
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    accelerator.print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Load datasets
    try:
        with accelerator.main_process_first():
            dataset = load_dataset(cfg.dataset_name, split=cfg.split)
            accelerator.print(f"Loaded training dataset: {len(dataset)} samples")
            
            # Load or create validation dataset
            val_dataset = None
            try:
                val_dataset = load_dataset(cfg.dataset_name, split=f"{cfg.val_split}[:{int(cfg.val_size*100)}%]")
                accelerator.print(f"Loaded validation dataset: {len(val_dataset)} samples")
            except:
                # Create validation split from training data
                dataset_split = dataset.train_test_split(test_size=cfg.val_size, seed=cfg.seed)
                dataset = dataset_split["train"]
                val_dataset = dataset_split["test"]
                accelerator.print(f"Created validation split: {len(val_dataset)} samples")
    
    except Exception as e:
        accelerator.print(f"Error loading dataset: {e}")
        raise
    
    # Tokenization function
    def tokenize_function(examples):
        # Handle different text field names
        text_key = None
        for key in ["text", "story", "content"]:
            if key in examples:
                text_key = key
                break
        
        if text_key is None:
            text_key = list(examples.keys())[0]
        
        texts = examples[text_key]
        if isinstance(texts[0], list):
            texts = [" ".join(text) if isinstance(text, list) else str(text) for text in texts]
        
        # Tokenize with proper truncation
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
            return_attention_mask=False,
        )
        
        # Filter out sequences that are too short
        filtered_input_ids = []
        for ids in tokenized["input_ids"]:
            if len(ids) >= 10:  # Minimum sequence length
                filtered_input_ids.append(ids)
        
        return {"input_ids": filtered_input_ids}
    
    # Apply tokenization
    with accelerator.main_process_first():
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=4,
            remove_columns=dataset.column_names,
            desc="Tokenizing training data"
        )
        
        if val_dataset is not None:
            val_dataset = val_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=1000,
                num_proc=4,
                remove_columns=val_dataset.column_names,
                desc="Tokenizing validation data"
            )
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=cfg.seed)
    
    return tokenizer, dataset, val_dataset

def create_data_collator(tokenizer):
    """Create data collator with robust error handling"""
    
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Filter valid samples
        valid_samples = []
        for item in batch:
            if "input_ids" in item and len(item["input_ids"]) > 0:
                valid_samples.append(item["input_ids"])
        
        if not valid_samples:
            # Return empty batch - will be skipped
            return None
        
        # Convert to tensors and pad
        input_ids = [torch.tensor(ids, dtype=torch.long) for ids in valid_samples]
        max_len = max(seq.size(0) for seq in input_ids)
        
        # Create padded tensors
        padded_input_ids = torch.full((len(input_ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(input_ids), max_len), dtype=torch.long)
        
        for i, seq in enumerate(input_ids):
            seq_len = seq.size(0)
            padded_input_ids[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1
        
        # Create labels (shifted input_ids with padding masked)
        labels = padded_input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    return collate_fn

def setup_model_and_optimizer(cfg, accelerator, vocab_size):
    """Setup model and optimizer with proper configuration"""
    
    # Import model classes - they should be available in scope
    assert 'ValkyrieConfig' in globals() and 'ValkyrieModel' in globals(), \
        "ValkyrieConfig and ValkyrieModel must be defined"
    
    # Create model config
    model_config = ValkyrieConfig(
        vocab_size=vocab_size,
        **cfg.model_config
    )
    
    # Create model
    model = ValkyrieModel(model_config)
    
    # Print model info
    model_size = get_model_size(model)
    accelerator.print(f"Created model with {model_size:.1f}M parameters")
    
    # Estimate memory usage
    est_memory = estimate_memory_usage(model, cfg.per_device_batch_size, cfg.max_length)
    accelerator.print(f"Estimated memory usage: {est_memory:.1f} GB")
    
    # Enable gradient checkpointing
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        accelerator.print("Gradient checkpointing enabled")
    
    # Model compilation (PyTorch 2.0)
    if cfg.compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            accelerator.print("Model compiled with torch.compile")
        except Exception as e:
            accelerator.print(f"Failed to compile model: {e}")
    
    # Setup optimizer with parameter grouping
    no_decay = ["bias", "LayerNorm.weight", "norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=cfg.learning_rate,
        betas=cfg.adam_betas,
        eps=cfg.adam_epsilon,
    )
    
    return model, optimizer, model_config

def setup_training_loop(cfg, accelerator, model, optimizer, train_dataloader, val_dataloader=None):
    """Setup scheduler and training state"""
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    
    # Setup scheduler
    num_warmup_steps = int(cfg.warmup_ratio * cfg.max_train_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=cfg.max_train_steps
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)
    
    accelerator.print(f"Training setup complete:")
    accelerator.print(f"  Total steps: {cfg.max_train_steps}")
    accelerator.print(f"  Warmup steps: {num_warmup_steps}")
    accelerator.print(f"  Effective batch size: {cfg.per_device_batch_size * cfg.gradient_accumulation_steps * accelerator.num_processes}")
    
    return model, optimizer, scheduler, train_dataloader, val_dataloader

# -------------------------
# Training Functions
# -------------------------
def evaluate_model(model, val_dataloader, accelerator):
    """Evaluate model on validation set"""
    if val_dataloader is None:
        return None
    
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            if batch is None:
                continue
            
            try:
                # Forward pass using model's built-in loss calculation
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True
                )
                
                loss = outputs["loss"]
                
                # Gather loss across processes
                loss = accelerator.gather(loss)
                total_loss += loss.mean().item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)
                
            except Exception as e:
                accelerator.print(f"Error in validation batch: {e}")
                continue
    
    model.train()
    
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        perplexity = math.exp(min(avg_loss, 20))  # Cap to prevent overflow
        return {"val_loss": avg_loss, "val_perplexity": perplexity}
    
    return None

def save_checkpoint(accelerator, cfg, global_step, is_best=False):
    """Save checkpoint with error handling"""
    try:
        if is_best:
            save_path = Path(cfg.output_dir) / "checkpoint-best"
        else:
            save_path = Path(cfg.output_dir) / f"checkpoint-step-{global_step}"
        
        accelerator.save_state(str(save_path))
        accelerator.print(f"Checkpoint saved: {save_path}")
        
        # Cleanup old checkpoints
        if not is_best and cfg.save_total_limit > 0:
            checkpoints = sorted([
                p for p in Path(cfg.output_dir).glob("checkpoint-step-*") 
                if p.is_dir()
            ])
            
            while len(checkpoints) > cfg.save_total_limit:
                old_checkpoint = checkpoints.pop(0)
                import shutil
                shutil.rmtree(old_checkpoint)
                accelerator.print(f"Removed old checkpoint: {old_checkpoint}")
        
        return True
        
    except Exception as e:
        accelerator.print(f"Failed to save checkpoint: {e}")
        return False

# -------------------------
# Main Training Loop
# -------------------------
def main():
    # Setup
    cfg, accelerator = setup_training()
    accelerator.print("Starting Valkyrie training on TinyStories")
    
    # Create output directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.logging_dir, exist_ok=True)
    
    # Setup data
    tokenizer, train_dataset, val_dataset = setup_tokenizer_and_data(cfg, accelerator)
    collate_fn = create_data_collator(tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.dataloader_pin_memory,
        drop_last=cfg.dataloader_drop_last,
    )
    
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.per_device_batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.dataloader_pin_memory,
            drop_last=False,
        )
    
    # Setup model and optimizer
    model, optimizer, model_config = setup_model_and_optimizer(cfg, accelerator, tokenizer.vocab_size)
    
    # Setup training loop components
    model, optimizer, scheduler, train_dataloader, val_dataloader = setup_training_loop(
        cfg, accelerator, model, optimizer, train_dataloader, val_dataloader
    )
    
    # Training state
    global_step = 0
    best_val_loss = float("inf")
    early_stopping_counter = 0
    training_start_time = time.time()
    
    # Resume from checkpoint if specified
    if cfg.resume_from_checkpoint:
        accelerator.load_state(cfg.resume_from_checkpoint)
        # Extract global step from checkpoint path if possible
        if "step" in cfg.resume_from_checkpoint:
            try:
                global_step = int(cfg.resume_from_checkpoint.split("step-")[-1])
            except ValueError:
                pass
        accelerator.print(f"Resumed from checkpoint: {cfg.resume_from_checkpoint}")
    
    # Initialize tracking
    if cfg.report_to:
        accelerator.init_trackers("valkyrie-training", config=cfg.__dict__)
    
    try:
        # Training loop
        model.train()
        accelerator.print("Starting training...")
        
        for epoch in range(cfg.num_train_epochs):
            accelerator.print(f"Epoch {epoch + 1}/{cfg.num_train_epochs}")
            
            for step, batch in enumerate(train_dataloader):
                if batch is None:
                    continue
                
                with accelerator.accumulate(model):
                    # Forward pass with built-in loss calculation
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        return_dict=True
                    )
                    
                    loss = outputs["loss"]
                    
                    # Check for NaN/Inf loss
                    if not torch.isfinite(loss):
                        accelerator.print(f"Warning: Non-finite loss at step {global_step}: {loss.item()}")
                        continue
                    
                    # Backward pass
                    accelerator.backward(loss)
                    
                    # Check model health
                    if global_step % 1000 == 0:
                        health_issues = check_model_health(accelerator.unwrap_model(model))
                        if health_issues:
                            accelerator.print(f"Model health issues: {health_issues}")
                    
                    # Optimizer step
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    if accelerator.sync_gradients:
                        global_step += 1
                
                # Logging
                if global_step % cfg.log_interval == 0:
                    lr = scheduler.get_last_lr()[0]
                    
                    logs = {
                        "train/loss": loss.item(),
                        "train/learning_rate": lr,
                        "train/step": global_step,
                        "train/epoch": epoch + 1,
                    }
                    
                    accelerator.log(logs, step=global_step)
                    accelerator.print(f"Step {global_step} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
                
                # Validation
                if global_step % cfg.eval_steps == 0:
                    val_metrics = evaluate_model(model, val_dataloader, accelerator)
                    if val_metrics:
                        val_logs = {f"val/{k}": v for k, v in val_metrics.items()}
                        accelerator.log(val_logs, step=global_step)
                        accelerator.print(f"Validation | Loss: {val_metrics['val_loss']:.4f} | PPL: {val_metrics['val_perplexity']:.2f}")
                        
                        # Early stopping and best model saving
                        if val_metrics["val_loss"] < best_val_loss:
                            best_val_loss = val_metrics["val_loss"]
                            early_stopping_counter = 0
                            save_checkpoint(accelerator, cfg, global_step, is_best=True)
                        else:
                            early_stopping_counter += 1
                            if early_stopping_counter >= cfg.early_stopping_patience:
                                accelerator.print("Early stopping triggered")
                                break
                
                # Regular checkpointing
                if global_step % cfg.save_every_n_steps == 0:
                    save_checkpoint(accelerator, cfg, global_step, is_best=False)
                
                # Check if max steps reached
                if cfg.max_train_steps and global_step >= cfg.max_train_steps:
                    break
            
            # Early stopping check (break outer loop)
            if early_stopping_counter >= cfg.early_stopping_patience:
                break
            
            if cfg.max_train_steps and global_step >= cfg.max_train_steps:
                break
    
    except KeyboardInterrupt:
        accelerator.print("Training interrupted by user")
    except Exception as e:
        accelerator.print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save final checkpoint
        try:
            final_path = Path(cfg.output_dir) / "checkpoint-final"
            accelerator.save_state(str(final_path))
            accelerator.print(f"Final checkpoint saved: {final_path}")
        except Exception as e:
            accelerator.print(f"Failed to save final checkpoint: {e}")
        
        # Training summary
        total_time = time.time() - training_start_time
        accelerator.print(f"Training completed in {total_time:.1f}s")
        accelerator.print(f"Total steps: {global_step}")
        accelerator.print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Cleanup tracking
        if cfg.report_to:
            accelerator.end_training()

if __name__ == "__main__":
    main()