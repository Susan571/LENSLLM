from typing import Dict, Optional, Union, List, Any, Tuple
from dataclasses import dataclass, field
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer, LlamaForCausalLM, TrainerCallback, TrainerControl, TrainerState
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, deepspeed_init

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import Repository, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.dependency_versions_check import dep_version_check
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_less_than_1_11
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_dataloader_sampler,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.trainer_utils import (

    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
from transformers.utils.quantization_config import QuantizationMethod


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch


if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin

    if version.parse(accelerate_version) > version.parse("0.20.3"):
        from accelerate.utils import (
            load_fsdp_model,
            load_fsdp_optimizer,
            save_fsdp_model,
            save_fsdp_optimizer,
        )

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


if TYPE_CHECKING:
    import optuna


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

@dataclass
class ModelArguments:
    model_name: Optional[str] = field(default="acrastt/Marx-3B-V2", metadata={"help": "Name of the model."})
    max_length: int = field(default=512, metadata={"help": "Max length of the input sequence."})

@dataclass
class DataArguments:
    dataset_name: str = field(default="casehold", metadata={"help": "Name of the data."})
    num_train_samples: int = field(default=-1, metadata={"help": "Number of samples for training."})

@dataclass
class EvaluationArguments:
    eval_num_beams: int = field(default=1, metadata={"help": "Number of beams for evaluation."})
    eval_max_new_length: int = field(default=5, metadata={"help": "Max length of the new tokens of generation for evaluation."})
    temperature: float = field(default=0.0, metadata={"help": "Temperature for generation."})
    eval_batch_size: int = field(default=8, metadata={"help": "Batch size for evaluation."})
    eval_steps: int = field(default=100, metadata={"help": "Number of steps for evaluation."})
    use_fschat: bool = field(default=False, metadata={"help": "Whether to use fastchat for evaluation."})
    use_parallel: int = field(default=1, metadata={"help": "Number of device for generation"})
    delete_ckpt: bool = field(default=True, metadata={"help": "Whether to delete the checkpoint after evaluation."})
    model_seed: int = field(default=42, metadata={"help": "Random seed for model."})
    num_demonstrations: int = field(default=0, metadata={"help": "Number of demonstrations for in-context learning."})
    do_sample: bool = field(default=False, metadata={"help": "Whether to sample for generation."})

@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    num_devices: int = field(default=None, metadata={"help": "Number of devices."})
    total_batch_size: int = field(default=None, metadata={"help": "Total training batch size."})



class MySeq2SeqTrainer(Seq2SeqTrainer):
    # def prediction_step(
    #     self,
    #     model: nn.Module,
    #     inputs: Dict[str, Union[torch.Tensor, Any]],
    #     prediction_loss_only: bool,
    #     ignore_keys: Optional[List[str]] = None,
    #     **gen_kwargs,
    # ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     """
    #     Perform an evaluation step on `model` using `inputs`.

    #     Subclass and override to inject custom behavior.

    #     Args:
    #         model (`nn.Module`):
    #             The model to evaluate.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.

    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.
    #         prediction_loss_only (`bool`):
    #             Whether or not to return the loss only.
    #         gen_kwargs:
    #             Additional `generate` specific kwargs.

    #     Return:
    #         Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
    #         labels (each being optional).
    #     """

    #     if not self.args.predict_with_generate or prediction_loss_only:
    #         return super().prediction_step(
    #             model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
    #         )

    #     has_labels = "labels" in inputs
    #     inputs = self._prepare_inputs(inputs)

    #     # XXX: adapt synced_gpus for fairscale as well
    #     # Priority (handled in generate):
    #     # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
    #     if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
    #         gen_kwargs = self._gen_kwargs.copy()
    #     # gen_kwargs = {'max_length': self.args.generation_max_length,
    #     #               'num_beams': self.args.generation_num_beams}
    #     # print(gen_kwargs) # DEBUG
    #     if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
    #         gen_kwargs.pop("num_beams")
    #     if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
    #         gen_kwargs.pop("max_length")

    #     default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
    #     gen_kwargs["synced_gpus"] = (
    #         gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
    #     )

    #     generation_inputs = inputs.copy()
    #     generation_inputs['input_ids'] = inputs['generation_input_ids']
    #     generation_inputs.pop('generation_input_ids')
    #     generation_inputs.pop('attention_mask')
    #     # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
    #     # (otherwise, it would continue generating from the padded `decoder_input_ids`)
    #     if (
    #         "labels" in generation_inputs
    #         and "decoder_input_ids" in generation_inputs
    #         and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
    #     ):
    #         generation_inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
    #     # print(gen_kwargs) #DEBUG
    #     generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

    #     # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
    #     # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
    #     # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
    #     if self.model.generation_config._from_model_config:
    #         self.model.generation_config._from_model_config = False

    #     # Retrieves GenerationConfig from model.generation_config
    #     gen_config = self.model.generation_config
    #     # in case the batch is shorter than max length, the output should be padded
    #     if generated_tokens.shape[-1] < gen_config.max_length:
    #         generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
    #     elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
    #         generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

    #     inputs = inputs.copy()
    #     inputs.pop('generation_input_ids')
    #     with torch.no_grad():
    #         if has_labels:
    #             with self.compute_loss_context_manager():
    #                 outputs = model(**inputs)
    #             if self.label_smoother is not None:
    #                 loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
    #             else:
    #                 loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
    #         else:
    #             loss = None

    #     if self.args.prediction_loss_only:
    #         return loss, None, None

    #     if has_labels:
    #         labels = inputs["labels"]
    #         if labels.shape[-1] < gen_config.max_length:
    #             labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
    #         elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
    #             labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
    #     else:
    #         labels = None

    #     return loss, generated_tokens, labels



    # def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
    #     """
    #     Returns the evaluation [`~torch.utils.data.DataLoader`].

    #     Subclass and override this method if you want to inject some custom behavior.

    #     Args:
    #         eval_dataset (`torch.utils.data.Dataset`, *optional*):
    #             If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
    #             by the `model.forward()` method are automatically removed. It must implement `__len__`.
    #     """
    #     if eval_dataset is None and self.eval_dataset is None:
    #         raise ValueError("Trainer: evaluation requires an eval_dataset.")
    #     eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
    #     data_collator = self.data_collator

    #     # No need to remove unused columns in eval dataset
    #     # if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
    #     #     eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
    #     # else:
    #     #     data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

    #     dataloader_params = {
    #         "batch_size": self.args.eval_batch_size,
    #         "collate_fn": data_collator,
    #         "num_workers": self.args.dataloader_num_workers,
    #         "pin_memory": self.args.dataloader_pin_memory,
    #     }

    #     if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
    #         dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
    #         dataloader_params["drop_last"] = self.args.dataloader_drop_last

    #     return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    # def evaluation_loop(
    #     self,
    #     dataloader: DataLoader,
    #     description: str,
    #     prediction_loss_only: Optional[bool] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ) -> EvalLoopOutput:
    #     """
    #     Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

    #     Works both with or without labels.
    #     """
    #     args = self.args

    #     prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

    #     # if eval is called w/o train, handle model prep here
    #     if self.is_deepspeed_enabled and self.deepspeed is None:
    #         _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

    #     model = self._wrap_model(self.model, training=False, dataloader=dataloader)

    #     if len(self.accelerator._models) == 0 and model is self.model:
    #         model = (
    #             self.accelerator.prepare(model)
    #             if self.is_deepspeed_enabled
    #             else self.accelerator.prepare_model(model, evaluation_mode=True)
    #         )

    #         if self.is_fsdp_enabled:
    #             self.model = model

    #         # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #         if model is not self.model:
    #             self.model_wrapped = model

    #         # backward compatibility
    #         if self.is_deepspeed_enabled:
    #             self.deepspeed = self.model_wrapped

    #     # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
    #     # while ``train`` is running, cast it to the right dtype first and then put on device
    #     if not self.is_in_train:
    #         if args.fp16_full_eval:
    #             model = model.to(dtype=torch.float16, device=args.device)
    #         elif args.bf16_full_eval:
    #             model = model.to(dtype=torch.bfloat16, device=args.device)

    #     batch_size = self.args.eval_batch_size

    #     logger.info(f"***** Running {description} *****")
    #     if has_length(dataloader):
    #         logger.info(f"  Num examples = {self.num_examples(dataloader)}")
    #     else:
    #         logger.info("  Num examples: Unknown")
    #     logger.info(f"  Batch size = {batch_size}")

    #     model.eval()

    #     self.callback_handler.eval_dataloader = dataloader
    #     # Do this before wrapping.
    #     eval_dataset = getattr(dataloader, "dataset", None)

    #     if args.past_index >= 0:
    #         self._past = None

    #     # Initialize containers
    #     # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
    #     losses_host = None
    #     preds_host = None
    #     labels_host = None
    #     inputs_host = None

    #     # losses/preds/labels on CPU (final containers)
    #     all_losses = None
    #     all_preds = None
    #     all_labels = None
    #     all_inputs = None
    #     # Will be useful when we have an iterable dataset so don't know its length.

    #     observed_num_examples = 0
    #     # Main evaluation loop
    #     for step, inputs in enumerate(dataloader):
    #         # Update the observed num examples
    #         observed_batch_size = find_batch_size(inputs)
    #         if observed_batch_size is not None:
    #             observed_num_examples += observed_batch_size
    #             # For batch samplers, batch_size is not known by the dataloader in advance.
    #             if batch_size is None:
    #                 batch_size = observed_batch_size

    #         # Prediction step
    #         loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
    #         main_input_name = getattr(self.model, "main_input_name", "input_ids")
    #         inputs_decode = self._prepare_input(inputs['generation_input_ids']) if args.include_inputs_for_metrics else None

    #         # Update containers on host
    #         if loss is not None:
    #             losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
    #             losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
    #         if labels is not None:
    #             labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
    #         if inputs_decode is not None:
    #             inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
    #             inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
    #             inputs_host = (
    #                 inputs_decode
    #                 if inputs_host is None
    #                 else nested_concat(inputs_host, inputs_decode, padding_index=-100)
    #             )
    #         if logits is not None:
    #             logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
    #             if self.preprocess_logits_for_metrics is not None:
    #                 logits = self.preprocess_logits_for_metrics(logits, labels)
    #             logits = self.accelerator.gather_for_metrics((logits))
    #             preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

    #         if labels is not None:
    #             labels = self.accelerator.gather_for_metrics((labels))
    #             labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

    #         self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

    #         # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
    #         if (
    #             args.eval_accumulation_steps is not None
    #             and (step + 1) % args.eval_accumulation_steps == 0
    #             and (self.accelerator.sync_gradients or version.parse(accelerate_version) > version.parse("0.20.3"))
    #         ):
    #             if losses_host is not None:
    #                 losses = nested_numpify(losses_host)
    #                 all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #             if preds_host is not None:
    #                 logits = nested_numpify(preds_host)
    #                 all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #             if inputs_host is not None:
    #                 inputs_decode = nested_numpify(inputs_host)
    #                 all_inputs = (
    #                     inputs_decode
    #                     if all_inputs is None
    #                     else nested_concat(all_inputs, inputs_decode, padding_index=-100)
    #                 )
    #             if labels_host is not None:
    #                 labels = nested_numpify(labels_host)
    #                 all_labels = (
    #                     labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
    #                 )

    #             # Set back to None to begin a new accumulation
    #             losses_host, preds_host, inputs_host, labels_host = None, None, None, None

    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of the evaluation loop
    #         delattr(self, "_past")

    #     # Gather all remaining tensors and put them back on the CPU
    #     if losses_host is not None:
    #         losses = nested_numpify(losses_host)
    #         all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #     if preds_host is not None:
    #         logits = nested_numpify(preds_host)
    #         all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #     if inputs_host is not None:
    #         inputs_decode = nested_numpify(inputs_host)
    #         all_inputs = (
    #             inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
    #         )
    #     if labels_host is not None:
    #         labels = nested_numpify(labels_host)
    #         all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

    #     # Number of samples
    #     if has_length(eval_dataset):
    #         num_samples = len(eval_dataset)
    #     # The instance check is weird and does not actually check for the type, but whether the dataset has the right
    #     # methods. Therefore we need to make sure it also has the attribute.
    #     elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
    #         num_samples = eval_dataset.num_examples
    #     else:
    #         if has_length(dataloader):
    #             num_samples = self.num_examples(dataloader)
    #         else:  # both len(dataloader.dataset) and len(dataloader) fail
    #             num_samples = observed_num_examples
    #     if num_samples == 0 and observed_num_examples > 0:
    #         num_samples = observed_num_examples

    #     # Metrics!
    #     if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
    #         if args.include_inputs_for_metrics:
    #             metrics = self.compute_metrics(
    #                 EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
    #             )
    #         else:
    #             metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    #     else:
    #         metrics = {}

    #     # To be JSON-serializable, we need to remove numpy types or zero-d tensors
    #     metrics = denumpify_detensorize(metrics)

    #     if all_losses is not None:
    #         metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
    #     if hasattr(self, "jit_compilation_time"):
    #         metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

    #     # Prefix all keys with metric_key_prefix + '_'
    #     for key in list(metrics.keys()):
    #         if not key.startswith(f"{metric_key_prefix}_"):
    #             metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

    #     return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    # def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    #     """
    #     Will save the model, so you can reload it using `from_pretrained()`.

    #     Will only save from the main process.
    #     """

    #     if output_dir is None:
    #         output_dir = self.args.output_dir

    #     if is_sagemaker_mp_enabled():
    #         # Calling the state_dict needs to be done on the wrapped model and on all processes.
    #         os.makedirs(output_dir, exist_ok=True)
    #         state_dict = self.model_wrapped.state_dict()
    #         if self.args.should_save:
    #             self._save(output_dir, state_dict=state_dict)
    #         if IS_SAGEMAKER_MP_POST_1_10:
    #             # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
    #             Path(os.path.join(output_dir, "user_content.pt")).touch()
    #     elif (
    #         ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
    #         or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
    #         or self.fsdp is not None
    #         or self.is_fsdp_enabled
    #     ):
    #         state_dict = self.model.state_dict() if not self.is_fsdp_enabled else {}
    #         if self.args.should_save:
    #             self._save(output_dir, state_dict=state_dict)
    #         if self.is_fsdp_enabled:
    #             # remove the dummy state_dict
    #             remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
    #             save_fsdp_model(self.accelerator.state.fsdp_plugin, self.accelerator, self.model, output_dir)

    #     elif self.is_deepspeed_enabled:
    #         # this takes care of everything as long as we aren't under zero3
    #         if version.parse(accelerate_version) <= version.parse("0.20.3"):
    #             raise ValueError("Install Accelerate from main branch")
    #         try:
    #             state_dict = self.accelerator.get_state_dict(self.deepspeed)
    #             if self.args.should_save:
    #                 self._save(output_dir, state_dict=state_dict)
    #         except ValueError:
    #             logger.warning(
    #                 " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
    #                 " zero_to_fp32.py to recover weights"
    #             )
    #             if self.args.should_save:
    #                 self._save(output_dir, state_dict={})
    #             # remove the dummy state_dict
    #             remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
    #             self.model_wrapped.save_checkpoint(output_dir)

    #     elif self.args.should_save:
    #         self._save(output_dir)

    #     # Push to the Hub when `save_model` is called by the user.
    #     if self.args.push_to_hub and not _internal_call:
    #         self.push_to_hub(commit_message="Model save")


    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        # MODIFICATION: don't save deepspeed ckpt (including model states & optimizer states)
        # if self.is_deepspeed_enabled:
        #     # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
        #     # config `stage3_gather_16bit_weights_on_model_save` is True
        #     self.model_wrapped.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if self.fsdp or self.is_fsdp_enabled:
            if self.is_fsdp_enabled:
                save_fsdp_optimizer(
                    self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, output_dir
                )
            else:
                # FSDP has a different interface for saving optimizer states.
                # Needs to be called on all ranks to gather all states.
                # full_optim_state_dict will be deprecated after Pytorch 2.2!
                full_osd = self.model.__class__.full_optim_state_dict(self.model, self.optimizer)
                torch.save(full_osd, os.path.join(output_dir, OPTIMIZER_NAME))

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
        elif self.args.should_save and not self.is_deepspeed_enabled and not (self.fsdp or self.is_fsdp_enabled):
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

        # Save SCHEDULER & SCALER
        is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and not isinstance(
            self.lr_scheduler, DeepSpeedSchedulerWrapper
        )
        if (
            self.args.should_save
            and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler)
            and not is_torch_tpu_available()
        ):
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

class RemoveZeroCkptCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        global_step_files = glob.glob(os.path.join(args.output_dir, "checkpoint-*/global_step*"))
        if state.is_local_process_zero:
            os.system(f"rm -rf {' '.join(global_step_files)}")
            print("Remove global_step files")
            print(global_step_files)

class EarlyStoppingCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles early stopping.

    Args:
       early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
       early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `

    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality to set best_metric
    in [`TrainerState`]. Note that if the [`TrainingArguments`] argument *save_steps* differs from *eval_steps*, the
    early stopping will not occur until the next save step.
    """

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        #assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        assert (
            args.metric_for_best_model is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        assert (
            args.evaluation_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True