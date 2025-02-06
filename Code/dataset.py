from dataclasses import dataclass
import pathos.multiprocessing as mp
from itertools import chain
import os
import json
import csv
import copy
import logging
import random
from typing import Dict, Optional, Sequence, Union

from torch.utils.data import Dataset
import torch
import transformers
from transformers import AutoTokenizer
import datasets

from utils.const_utils import *
from utils.env_utils import *


def load_dataset(tokenizer: transformers.PreTrainedTokenizer, model_args, data_args, split='train', num_demos=0) -> Dict:
    """Make dataset and collator for supervised fine-tuning.
    Args:
    - tokenizer: A `PreTrainedTokenizer` object.
    - model_args: A `ModelArguments` object. Used for choosing specific instruction format.
    - data_args: A `DataArguments` object.
    - split: str, one of 'train', 'valid', 'test'.
    - num_demos: int, number of demonstrations to use for in-context learning.
    Return a Dict containing:
    - train_dataset: A `SupervisedDataset` object.
    - eval_dataset: A `SupervisedDataset` object.
    - data_collator
    """
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if split == 'test':
        test_dataset = SupervisedDataset(tokenizer=tokenizer, dataset_name=data_args.dataset_name, split='test', num_demos=num_demos)
        return dict(eval_dataset=test_dataset, data_collator=data_collator)
    elif split == 'valid':
        valid_dataset = SupervisedDataset(tokenizer=tokenizer, dataset_name=data_args.dataset_name, split='valid', num_demos=num_demos)
        return dict(eval_dataset=valid_dataset, data_collator=data_collator) 
    else: 
        train_dataset = SupervisedDataset(tokenizer=tokenizer, dataset_name=data_args.dataset_name, split='train', num_samples=data_args.num_train_samples, num_demos=num_demos, seed=data_args.seed)
        valid_dataset = SupervisedDataset(tokenizer=tokenizer, dataset_name=data_args.dataset_name, split='valid', num_demos=num_demos)
        return dict(train_dataset=train_dataset, eval_dataset=valid_dataset, data_collator=data_collator)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels  = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        forward_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ) # <s> x1, ..., xn, y1, ..., ym, </s> [PAD] ... [PAD]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        if "generation_input_ids" in instances[0]:
            generation_input_ids = [instance["generation_input_ids"] for instance in instances]
            self.tokenizer.padding_side = 'left'
            generation_input_ids = self.tokenizer.pad({'input_ids': generation_input_ids}, return_tensors='pt', padding=True).input_ids # [PAD] ... [PAD] <s> x1, ..., xn
            self.tokenizer.padding_side = 'right'
            return dict(
                input_ids=forward_input_ids,
                labels=labels,
                attention_mask=forward_input_ids.ne(self.tokenizer.pad_token_id),
                generation_input_ids=generation_input_ids
            )

        return dict(
            input_ids=forward_input_ids,
            labels=labels,
            attention_mask=forward_input_ids.ne(self.tokenizer.pad_token_id),
        )


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, 
                 lengths: Union[Sequence[int], int]) -> Dict:
    """Tokenize a list of strings."""
    if isinstance(lengths, int):
        lengths = [lengths] * len(strings)
    
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=l,
            truncation=True,
            add_special_tokens=False
        )
        for text, l in zip(strings, lengths)
    ]
    # fn = lambda args: tokenizer(
    #         args[0],
    #         return_tensors="pt",
    #         padding="longest",
    #         max_length=args[1],
    #         truncation=True,
    #         add_special_tokens=False
    #     )
    # with mp.Pool(8) as pool:
    #     tokenized_list = pool.map(
    #     fn, zip(strings, lengths))
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        input_ids_lens_without_special_tokens=input_ids_lens,
    )

def preprocess_with_demonstration(
    sources: Sequence[str],
    instructions: Sequence[str],
    targets: Sequence[str],
    demos: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    is_eval: bool=True
) -> Dict:
    """"""
    instructions_tokenized = _tokenize_fn(instructions, tokenizer, lengths=tokenizer.model_max_length // 2)
    targets_tokenized = _tokenize_fn(targets, tokenizer, lengths=tokenizer.model_max_length)
    demos_tokenized = _tokenize_fn(demos, tokenizer, lengths=tokenizer.model_max_length // 2)
    split_token_id = tokenizer(['\n'], add_special_tokens=False, return_tensors='pt').input_ids[0]
    demo_input_ids = torch.concat([i for input_ids in demos_tokenized['input_ids'] for i in (input_ids, split_token_id)])
    demo_length = sum(demos_tokenized['input_ids_lens_without_special_tokens'])
    
    sources_max_length = [tokenizer.model_max_length - len1 - len2 - demo_length - 2 
                            for len1, len2 in zip(targets_tokenized['input_ids_lens_without_special_tokens'], 
                                                        instructions_tokenized['input_ids_lens_without_special_tokens'])]
    assert all([l > 0 for l in sources_max_length]), "Some targets are too long!"

    sources_tokenized = _tokenize_fn(sources, tokenizer, lengths=sources_max_length)
    input_ids = [torch.concatenate((torch.tensor([tokenizer.bos_token_id]), 
                                    demo_input_ids,
                                    src,
                                    inst,
                                    tgt, 
                                    torch.tensor([tokenizer.eos_token_id]))) 
                                    for src, tgt, inst in zip(sources_tokenized['input_ids'],
                                                                    targets_tokenized['input_ids'],
                                                                    instructions_tokenized['input_ids'])]
    # input_ids:  <s> x1, ..., xn, y1, ..., ym, </s>
    labels = copy.deepcopy(input_ids)
    generation_input_ids = []
    for label, target_len, input_id in zip(labels, targets_tokenized['input_ids_lens_without_special_tokens'], input_ids):
        # Mask the source and instruction tokens
        label[:-1-target_len] = IGNORE_INDEX #  IGNORE ... IGNORE y1, ..., ym, </s>
        # Input_ids for generate() should not contain target tokens
        generation_input_id = input_id[:-1-target_len] # <s> x1, ..., xn
        generation_input_ids.append(copy.deepcopy(generation_input_id))
        
    return dict(input_ids=input_ids, labels=labels, generation_input_ids=generation_input_ids)



def preprocess_1(
    sources: Sequence[str],
    instructions: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    is_eval: bool=True
) -> Dict:
    """Preprocess the data by tokenizing.
    Will keep instructions and targets unmodified, and truncate sources from the right side.
    The final sequence will be: [bos] truncated_source instruction target [eos]
    """
    # # DEBUG
    # sources = sources[:50]
    # instructions = instructions[:50]
    # targets = targets[:50]

    instructions_tokenized = _tokenize_fn(instructions, tokenizer, lengths=tokenizer.model_max_length // 2)
    targets_tokenized = _tokenize_fn(targets, tokenizer, lengths=256)
    sources_tokenized = _tokenize_fn(sources, tokenizer, lengths=256)
    input_ids = [torch.concatenate((torch.tensor([tokenizer.bos_token_id]), 
                                    src,
                                    inst,
                                    tgt, 
                                    torch.tensor([tokenizer.eos_token_id]))) 
                                    for src, tgt, inst in zip(sources_tokenized['input_ids'],
                                                                targets_tokenized['input_ids'],
                                                                instructions_tokenized['input_ids'])]
    # input_ids:  <s> x1, ..., xn, y1, ..., ym, </s>
    labels = copy.deepcopy(input_ids)
    generation_input_ids = []
    for label, target_len, input_id in zip(labels, targets_tokenized['input_ids_lens_without_special_tokens'], input_ids):
        # Mask the source and instruction tokens
        label[:-1-target_len] = IGNORE_INDEX #  IGNORE ... IGNORE y1, ..., ym, </s>
        # Input_ids for generate() should not contain target tokens
        generation_input_id = input_id[:-1-target_len] # <s> x1, ..., xn
        generation_input_ids.append(copy.deepcopy(generation_input_id))
        
    return dict(input_ids=input_ids, labels=labels, generation_input_ids=generation_input_ids)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset_name: str, tokenizer: transformers.PreTrainedTokenizer, split: str, num_samples: int = -1, num_demos: int = 0, seed: int = 42):
        super(SupervisedDataset, self).__init__()

        # Load raw data
        logging.warning("Loading data...")
        raw_dataset_tuple = _load_dataset_from_src(dataset_name, split=split)

        # Load demonstrations if using in-context learning
        if num_demos > 0:
            logging.warning("Loading demonstrations...")
            demo_s, demo_i, demo_t, _, _ = _load_dataset_from_src(dataset_name, split='train')
            # use only the first two elements as demonstrations
            # TODO: buggy, maybe too long for some datasets
            demos = [s+i+t for s, i, t in zip(demo_s[:num_demos], demo_i[:num_demos], demo_t[:num_demos])]

        
        # Load limited data for training
        if split == 'train' and num_samples != -1:
            num_samples = min(num_samples, len(raw_dataset_tuple[0]))
            random.seed(seed)
            index_selected = random.sample(range(len(raw_dataset_tuple[0])), num_samples)
            raw_dataset_tuple = [[d[i] for i in index_selected] for d in raw_dataset_tuple]

        # Raw dataset (for evaluation)
        self.inputs = raw_dataset_tuple[3]
        self.outputs = raw_dataset_tuple[4]
        raw_dataset_tuple = raw_dataset_tuple[:3]

        # Preprocess
        logging.warning("Tokenizing...")
        if num_demos > 0:
            data_dict = preprocess_with_demonstration(*raw_dataset_tuple, demos=demos, tokenizer=tokenizer)
        else:
            data_dict = preprocess_1(*raw_dataset_tuple, tokenizer=tokenizer)

        # Tokenized dataset
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.generation_input_ids = data_dict["generation_input_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], 
                    labels=self.labels[i], 
                    generation_input_ids=self.generation_input_ids[i],
                    input=self.inputs[i],
                    output=self.outputs[i])

def _load_dataset_from_src(dataset_name, tokenizer=None, split='train'):
    """Dataset loading. When adding a new dataset, use [elif dataset_name == XXX] 
    
    Return a list of
    - sources: List[str], the source texts.
    - instructions: List[str], the instruction texts.
    - targets: List[str], the target texts.
    """
    # data_list = []

    if dataset_name == 'wmt19':
        if split == 'train':
            dataset = []
            for i in range(10):
                if i == 5 or i == 6: continue
                dataset.append(datasets.load_dataset('parquet', data_files={'train':os.path.join(DATA_DIR, 'wmt19', f'000{str(i)}.parquet')})['train'])
            dataset = datasets.concatenate_datasets(dataset)
            print(dataset)
            dataset = dataset['translation']
        else:
            dataset = datasets.load_dataset('parquet', data_files={'val':os.path.join(DATA_DIR, 'wmt19', 'val.parquet')})
            dataset = dataset['val']['translation']
        
        sources = []
        instructions = ["Translate to Chinese:"] * len(dataset)
        targets = []
        inputs = []
        outputs = []

        for data_point in dataset:
            sources.append('{}'.format(data_point['en']))
            targets.append("{}".format(data_point['zh']))
            inputs.append(sources[-1] + instructions[-1])
            outputs.append(targets[-1])
    else:
        raise NotImplementedError
    
    return sources, instructions, targets, inputs, outputs
    # return data_list

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained(f'{MODEL_DIR}/alpaca-7b')
    a = SupervisedDataset(dataset_name='sql', tokenizer=tokenizer, split='train', num_samples=1000, num_demos=0)
    train_dataset = _load_dataset_from_src(dataset_name='sql', tokenizer=tokenizer, split='train')
    # whole_input_ids = [i+o for i, o in zip(train_dataset[3], train_dataset[4])]
    # tokenized_whole_input_ids = [tokenizer.encode(i) for i in whole_input_ids]
    # tokenized_length = [len(i) for i in tokenized_whole_input_ids]
    # print(f"Avg length: {sum(tokenized_length) / len(tokenized_length)}, Max length: {max(tokenized_length)}")

    # print("load ok")
    # import pdb
    # pdb.set_trace()