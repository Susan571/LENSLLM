import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import wandb
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    HfArgumentParser
)
import torch.distributed as dist
import torch.autograd as autograd
from typing import Optional, Dict, Any, Tuple

# Local imports
from utils import (
    ModelArguments,
    DataArguments,
    MyTrainingArguments,
    EarlyStoppingCallback,
    MySeq2SeqTrainer,
    smart_tokenizer_and_embedding_resize,
    load_dataset,
    CKPT_DIR,
    MODEL_DIR,
    DEFAULT_TOKENS
)
from utils.env_utils import configure_environment

# Configure logging
logger = logging.getLogger(__name__)
configure_environment()

def initialize_model_and_tokenizer(
    model_args: ModelArguments,
    training_args: MyTrainingArguments
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Initialize model and tokenizer with proper configuration."""
    model_path = Path(MODEL_DIR)/model_args.model_name if MODEL_DIR else model_args.model_name
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
            device_map="auto" if training_args.deepspeed else None
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            model_max_length=model_args.max_length,
            use_fast="pythia" not in model_args.model_name,
            legacy=False
        )
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {str(e)}")
        raise

def handle_special_tokens(tokenizer: AutoTokenizer) -> Dict[str, str]:
    """Ensure required special tokens exist in the tokenizer."""
    special_tokens = {}
    for token_type, default_value in DEFAULT_TOKENS.items():
        if getattr(tokenizer, f"{token_type}_token") is None:
            special_tokens[token_type] = default_value
            logger.info(f"Adding {token_type} token: {default_value}")
    return special_tokens

def configure_training_arguments(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: MyTrainingArguments
) -> MyTrainingArguments:
    """Configure final training arguments with runtime adjustments."""
    # Set output directory structure
    training_args.output_dir = Path(CKPT_DIR).joinpath(
        model_args.model_name,
        data_args.dataset_name,
        f"seed{training_args.seed}",
        f"{data_args.num_train_samples}_samples"
    ).as_posix()

    # Configure mixed precision based on hardware support
    if torch.cuda.get_device_capability()[0] < 8:
        training_args.bf16 = False
        training_args.fp16 = True
    
    # Set common training parameters
    training_args.remove_unused_columns = True
    training_args.predict_with_generate = False
    training_args.include_inputs_for_metrics = False
    
    return training_args

class NTKTracker:
    """Track NTK matrix and compute NTK-based test loss during training"""
    def __init__(self, model: AutoModelForCausalLM, eta: float = 0.01):
        self.model = model
        self.eta = eta
        self.ntk_matrix = None
        self.f0_X = None
        self.early_stopping_time = None
        
    def compute_ntk_matrix(self, x, x_prime):
        """Compute NTK matrix according to equation (8)"""
        grad_x = autograd.grad(self.model(x).loss, self.model.parameters(), create_graph=True)
        grad_x_prime = autograd.grad(self.model(x_prime).loss, self.model.parameters(), create_graph=True)
        
        ntk = sum(torch.sum(g1 * g2) for g1, g2 in zip(grad_x, grad_x_prime))
        return ntk

    def compute_test_loss(self, y) -> float:
        """Compute NTK-based test loss according to equation (9)"""
        exp_term = torch.exp(-self.eta * self.ntk_matrix * self.early_stopping_time)
        diff = self.f0_X - y
        return torch.norm(torch.matmul(exp_term, diff), p=2) ** 2

class LensLLMTrainer(MySeq2SeqTrainer):
    """Custom trainer with NTK tracking and early stopping"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ntk_tracker = NTKTracker(self.model)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override to track NTK matrix during training"""
        if self.ntk_tracker.f0_X is None:
            # Store initial outputs
            self.ntk_tracker.f0_X = model(**inputs).logits.detach()
            
        # Compute NTK matrix for pre-training and fine-tuning features
        if self.ntk_tracker.ntk_matrix is None:
            x = inputs['input_ids']
            x_prime = inputs['input_ids']  # Use same inputs for simplification
            self.ntk_tracker.ntk_matrix = self.ntk_tracker.compute_ntk_matrix(x, x_prime)
            
        return super().compute_loss(model, inputs, return_outputs)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """Override to track early stopping time"""
        if self.state.global_step % self.args.logging_steps == 0:
            self.ntk_tracker.early_stopping_time = self.state.global_step
            
        return super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

def train() -> None:
    """Modified main training workflow with LensLLM components"""
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args = configure_training_arguments(model_args, data_args, training_args)
    data_args.seed = training_args.seed

    model, tokenizer = initialize_model_and_tokenizer(model_args, training_args)
    special_tokens = handle_special_tokens(tokenizer)
    smart_tokenizer_and_embedding_resize(special_tokens, tokenizer, model)

    data_module = load_dataset(
        tokenizer=tokenizer,
        model_args=model_args,
        data_args=data_args,
        split="train"
    )

    if Path(training_args.output_dir).exists():
        os.system(f"rm -rf {training_args.output_dir}")

    # Use LensLLMTrainer instead of MySeq2SeqTrainer
    trainer = LensLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[EarlyStoppingCallback(patience=3)],
        **data_module
    )

    # Train and collect NTK information
    trainer.train()
    
    # Save NTK-related metadata
    if dist.is_initialized() and dist.get_rank() == 0:
        ntk_metadata = {
            "early_stopping_time": trainer.ntk_tracker.early_stopping_time,
            "ntk_matrix": trainer.ntk_tracker.ntk_matrix.cpu().numpy().tolist(),
            "f0_X": trainer.ntk_tracker.f0_X.cpu().numpy().tolist()
        }
        save_training_metadata(training_args, ntk_metadata)

def save_training_metadata(training_args: MyTrainingArguments, ntk_metadata: Dict[str, Any]) -> None:
    """Save training and NTK metadata"""
    metadata = {
        "batch_size": training_args.total_batch_size,
        "num_devices": training_args.num_devices,
        "wandb_id": wandb.run.id if wandb.run else None,
        "ntk_data": ntk_metadata
    }
    
    try:
        output_file = Path(training_args.output_dir)/"run_metadata.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with output_file.open("w") as f:
            json.dump(metadata, f, indent=2)
            logger.info(f"Saved training metadata to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {str(e)}")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.critical(f"Training failed: {str(e)}")
        raise
