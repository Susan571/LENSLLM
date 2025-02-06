from typing import Dict, Optional, Union
import warnings
import torch
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict[str, Union[str, bool, int]],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    init_strategy: str = "mean",
    embedding_size_warning: bool = True,
) -> int:
    # Add special tokens and resize embeddings
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if num_new_tokens == 0:
        return 0  # Early return if no new tokens added

    model.resize_token_embeddings(len(tokenizer))
    
    # Warn about potential performance issues
    if embedding_size_warning:
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if embedding_size % 64 != 0:
            warnings.warn(
                f"Embedding size {embedding_size} isn't divisible by 64. "
                "This may lead to suboptimal GPU performance for some architectures.",
                UserWarning
            )

    # Initialize new embeddings using specified strategy
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = None
    
    if model.get_output_embeddings() is not None:
        output_embeddings = model.get_output_embeddings().weight.data

    # Slice indices for original embeddings
    original_embeddings = slice(None, -num_new_tokens) if num_new_tokens > 0 else slice(None)
    
    # Initialize new token embeddings
    for embeddings in [input_embeddings, output_embeddings]:
        if embeddings is None:
            continue
            
        if init_strategy == "mean":
            embeddings[-num_new_tokens:] = embeddings[original_embeddings].mean(dim=0)
        elif init_strategy == "zero":
            embeddings[-num_new_tokens:] = torch.zeros_like(embeddings[-num_new_tokens:])
        elif init_strategy == "small_normal":
            embeddings[-num_new_tokens:] = torch.randn_like(embeddings[-num_new_tokens:]) * 0.02
        else:
            raise ValueError(f"Invalid init_strategy: {init_strategy}")

    return num_new_tokens
