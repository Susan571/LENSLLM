from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import os
import glob
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch.utils.data import DataLoader, Dataset


@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="acrastt/Marx-3B-V2",
        metadata={"help": "Name or path of pretrained model"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for inputs"}
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default="casehold",
        metadata={"help": "Name of dataset to use"}
    )
    num_train_samples: int = field(
        default=-1,
        metadata={"help": "Number of training samples (-1 for full dataset)"}
    )


@dataclass 
class EvaluationArguments:
    eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for evaluation"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Run evaluation every X steps"}
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature for generation"}
    )
    model_seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )


@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
    num_devices: int = field(
        default=None,
        metadata={"help": "Number of accelerators available"}
    )
    total_batch_size: int = field(
        default=None,
        metadata={"help": "Total effective batch size across devices"}
    )


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """Custom trainer with modified checkpoint saving behavior"""
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """Save training checkpoint with custom handling for different parallel strategies"""
        checkpoint_dir = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_dir)
        
        # Save model weights
        self.save_model(output_dir, _internal_call=True)
        
        # Skip DeepSpeed checkpoint saving
        if self.is_deepspeed_enabled:
            logger.warning("Skipping DeepSpeed checkpoint save at user request")
            
        # Save optimizer/scheduler states
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()
            
        # Handle FSDP saving
        if self.fsdp or self.is_fsdp_enabled:
            self._save_fsdp_checkpoint(output_dir)
            
        # Save training state and metrics
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
            torch.save(self._get_rng_states(), os.path.join(output_dir, "rng_state.pth"))

        # Cleanup old checkpoints
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True)


class CheckpointCleanerCallback(TrainerCallback):
    """Callback to remove unnecessary checkpoint files"""
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Clean up global step files after checkpoint save"""
        if state.is_local_process_zero:
            step_files = glob.glob(os.path.join(args.output_dir, "checkpoint-*/global_step*"))
            for f in step_files:
                os.remove(f)
            logger.info(f"Cleaned {len(step_files)} temporary checkpoint files")


class EarlyStoppingCallback(TrainerCallback):
    """Implements early stopping based on validation metrics"""
    
    def __init__(self, patience: int = 3, threshold: float = 0.01):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best_metric = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Evaluate stopping condition after each validation"""
        metric_name = f"eval_{args.metric_for_best_model}"
        current_metric = metrics.get(metric_name)
        
        if current_metric is None:
            logger.warning(f"Early stopping metric {metric_name} not found")
            return
            
        if (self.best_metric is None or 
            (current_metric > (self.best_metric + self.threshold) if args.greater_is_better 
             else current_metric < (self.best_metric - self.threshold))):
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True
