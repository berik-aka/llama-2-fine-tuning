# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os

# import sys
# from typing import List, Union

import fire
import torch

# import transformers
# from datasets import load_dataset
# import os.path as osp
# from tqdm import tqdm

# Unused imports removed
from utils import fsdp_auto_wrap_policy
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    # AutoModelForSeq2SeqLM,
    # AutoTokenizer,
    default_data_collator,
    BitsAndBytesConfig,
)
import torch.distributed as dist

# Unused imports removed
from utils.train_utils import (
    # set_tokenizer_params,
    train,
    # evaluation,
    freeze_transformer_layers,
    # check_frozen_layers_peft_model,
    setup,
    setup_environ_flags,
    # cleanup,
    # clear_gpu_cache,
    # get_parameter_dtypes,
    print_model_size,
    get_policies,
)

from utils.dataset_utils import get_preprocessed_dataset

from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from peft import get_peft_model, TaskType, prepare_model_for_kbit_training
import configs
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.utils.data import DistributedSampler
import policies
from policies import AnyPrecisionAdamW
from configs import fsdp_config, train_config
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pkg_resources import packaging
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def main(**kwargs):
    # Update the configuration for the training and sharding process
    update_config(train_config, **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    # Calculate gradient accumulation steps
    gradient_accumulation_steps = (
        train_config.batch_size_training // train_config.micro_batch_size
    )

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # Load the pre-trained model and setup its configuration
    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        quantization_config=bnb_config,
        device_map="auto" if train_config.quantization else None,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(
        train_config.model_name, trust_remote_code=True
    )
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )
    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    dataset_config = generate_dataset_config(train_config, kwargs)

    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )

    train_sampler = None
    val_sampler = None

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=default_data_collator,
        )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=0.0,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        gradient_accumulation_steps,
        train_config,
    )
    [print(f"Key: {k}, Value: {v}") for k, v in results.items()]


if __name__ == "__main__":
    fire.Fire(main)
