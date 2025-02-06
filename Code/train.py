import os
from typing import Dict, Optional, Sequence, Union, List, Any, Tuple
import json

import torch
import wandb
import deepspeed
import transformers
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch.distributed as dist


from utils.env_utils import *
from utils.const_utils import *
from utils.custom_utils import ModelArguments, DataArguments, EarlyStoppingCallback, MySeq2SeqTrainer, MyTrainingArguments
from utils.func_utils import smart_tokenizer_and_embedding_resize
from dataset import load_dataset




def train():
    # Load args
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    # parser = deepspeed.add_config_arguments(parser)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if torch.cuda.get_device_capability()[0] < 8:
        training_args.bf16 = False
    print(training_args) #DEBUG
    training_args.remove_unused_columns = True
    training_args.output_dir = f"{CKPT_DIR}/{model_args.model_name}/{data_args.dataset_name}/seed{training_args.seed}/{data_args.num_train_samples}_samples"
    training_args.predict_with_generate = False
    training_args.include_inputs_for_metrics = False
    data_args.seed = training_args.seed
    # Load model and tokenizer
    model_path = f"{MODEL_DIR}/{model_args.model_name}" if MODEL_DIR else model_args.model_name
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        model_max_length=model_args.max_length,
        add_eos_token=True,
        use_fast=True if 'pythia' in model_args.model_name else False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    if tokenizer.cls_token is None:
        special_tokens_dict["cls_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.sep_token is None:
        special_tokens_dict["sep_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.mask_token is None:
        special_tokens_dict["mask_token"] = DEFAULT_MASK_TOKEN
        
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # Load dataset
    data_module = load_dataset(tokenizer=tokenizer, model_args=model_args, data_args=data_args, split="train")

    # Load metrics
    # compute_metrics = load_metrics(tokenizer=tokenizer, data_args=data_args)

    # Remove previous ckpt files before training
    if os.path.exists(training_args.output_dir):
        os.system(f"rm -rf {training_args.output_dir}")

    # Train
    trainer = MySeq2SeqTrainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], **data_module)
    # trainer = Seq2SeqTrainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=[RemoveZeroCkptCallback], **data_module)
    # trainer.evaluate()
    trainer.train()
    trainer.save_state()
    # trainer.save_model(output_dir=training_args.output_dir)
    # with open(f"{training_args.output_dir}/best_ckpt_path.json", "w") as f:
    #     json.dump({"best_ckpt_path": trainer.state.best_model_checkpoint}, f, indent=4)
    
    # save wandb run id into file in process 0
    if dist.get_rank() == 0:
        print("Saving wandb run id into file...")
        wandb.run.summary['batchsize'] = training_args.total_batch_size
        wandb.run.summary['num_devices'] = training_args.num_devices
        with open(f"{training_args.output_dir}/run_id.json", "w") as f:
            json.dump({'wandb_id': wandb.run.id}, f, indent=4)
    


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train()
    # evaluate()