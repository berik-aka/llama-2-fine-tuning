# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class musealpha_dataset:
    data_path: str
    dataset: str = "musealpha_dataset"
    train_split: str = "train"
    test_split: str = "test"
    input_length: int = 2048
