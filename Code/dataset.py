from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import logging
import os
import random
import copy
import json

import torch
from torch.utils.data import Dataset
import datasets
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizer

# Constants and environment configuration
from utils.const_utils import IGNORE_INDEX, DATA_DIR, DEFAULT_TOKENS
from utils.env_utils import configure_environment

configure_environment()
logger = logging.getLogger(__name__)

@dataclass
class DatasetComponents:
    sources: List[str]
    instructions: List[str]
    targets: List[str]
    inputs: List[str]
    outputs: List[str]

class DataCollatorForSupervisedDataset:
    """Collates batches for sequence-to-sequence training with optional generation inputs"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            key: torch.stack([torch.tensor(inst[key]) for inst in instances])
            for key in ("input_ids", "labels")
        }
        
        batch["attention_mask"] = batch["input_ids"].ne(self.tokenizer.pad_token_id)
        
        if "generation_input_ids" in instances[0]:
            self.tokenizer.padding_side = 'left'
            generation_ids = [inst["generation_input_ids"] for inst in instances]
            batch["generation_input_ids"] = self.tokenizer.pad(
                {"input_ids": generation_ids}, 
                return_tensors="pt"
            ).input_ids
            self.tokenizer.padding_side = 'right'
            
        return batch

def tokenize_batch(
    texts: Sequence[str], 
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """Tokenizes text batches with dynamic length handling"""
    max_length = max_length or tokenizer.model_max_length
    return tokenizer(
        texts,
        padding="longest",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False
    )

def build_sequence(
    tokenizer: PreTrainedTokenizer,
    *components: torch.Tensor,
    add_special_tokens: bool = True
) -> torch.Tensor:
    """Constructs a model input sequence from tokenized components"""
    sequence = []
    if add_special_tokens:
        sequence.append(torch.tensor([tokenizer.bos_token_id]))
    
    sequence.extend(components)
    
    if add_special_tokens:
        sequence.append(torch.tensor([tokenizer.eos_token_id]))
        
    return torch.cat(sequence)

def preprocess_samples(
    sources: Sequence[str],
    instructions: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
    demonstrations: Optional[Sequence[str]] = None
) -> Dict[str, List[torch.Tensor]]:
    """Preprocesses data samples with optional in-context demonstrations"""
    # Tokenize components
    instruction_tokens = tokenize_batch(instructions, tokenizer, tokenizer.model_max_length//2)
    target_tokens = tokenize_batch(targets, tokenizer)
    demo_tokens = tokenize_batch(demonstrations, tokenizer, tokenizer.model_max_length//2) if demonstrations else None

    # Calculate available space for sources
    source_max_lengths = [
        tokenizer.model_max_length - len(inst) - len(tgt) - (len(demo_tokens) if demo_tokens else 0) - 2
        for inst, tgt in zip(instruction_tokens, target_tokens)
    ]
    
    if any(l <= 0 for l in source_max_lengths):
        raise ValueError("Source text too long for model context window")

    # Tokenize sources with dynamic length constraints
    source_tokens = tokenize_batch(sources, tokenizer, max_length=source_max_lengths)

    # Construct full sequences
    sequences = []
    generation_inputs = []
    
    for src, inst, tgt in zip(source_tokens, instruction_tokens, target_tokens):
        components = [src, inst, tgt]
        if demonstrations:
            components.insert(0, demo_tokens)
            
        seq = build_sequence(tokenizer, *components)
        sequences.append(seq)
        
        # Create generation inputs (context without target)
        gen_input = build_sequence(tokenizer, src, inst, add_special_tokens=False)
        generation_inputs.append(gen_input)

    # Create label masks
    labels = [seq.clone() for seq in sequences]
    for label, target in zip(labels, target_tokens):
        label[:-len(target)-1] = IGNORE_INDEX  # Mask non-target tokens

    return {
        "input_ids": sequences,
        "labels": labels,
        "generation_input_ids": generation_inputs
    }

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with in-context learning support"""
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        num_samples: int = -1,
        num_demos: int = 0,
        seed: int = 42
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.split = split
        
        # Load and prepare data
        data = self._load_data(dataset_name, split, num_samples, seed)
        self.samples = self._preprocess_data(data, num_demos)
        
    def _load_data(
        self,
        dataset_name: str,
        split: str,
        num_samples: int,
        seed: int
    ) -> DatasetComponents:
        """Loads and samples raw dataset"""
        logger.info(f"Loading {split} data for {dataset_name}")
        components = load_dataset_from_source(dataset_name, split)
        
        if split == "train" and num_samples > 0:
            random.seed(seed)
            indices = random.sample(range(len(components.sources)), min(num_samples, len(components.sources)))
            return DatasetComponents(*[
                [c[i] for i in indices] for c in (
                    components.sources,
                    components.instructions,
                    components.targets,
                    components.inputs,
                    components.outputs
                )
            ])
            
        return components

    def _preprocess_data(
        self,
        data: DatasetComponents,
        num_demos: int
    ) -> Dict[str, List[torch.Tensor]]:
        """Handles data preprocessing with optional demonstrations"""
        demos = None
        if num_demos > 0:
            logger.info(f"Loading {num_demos} demonstrations")
            demo_data = load_dataset_from_source(self.dataset_name, "train")
            demos = [s+i+t for s, i, t in zip(
                demo_data.sources[:num_demos],
                demo_data.instructions[:num_demos],
                demo_data.targets[:num_demos]
            )]

        return preprocess_samples(
            sources=data.sources,
            instructions=data.instructions,
            targets=data.targets,
            tokenizer=self.tokenizer,
            demonstrations=demos
        )

    def __len__(self) -> int:
        return len(self.samples["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            key: torch.tensor(val[idx]) 
            for key, val in self.samples.items()
        }