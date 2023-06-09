# -*- coding: utf-8 -*-
r""" 
DataModule
==========
    The DataModule encapsulates all the steps needed to process data.
"""
import hashlib
import multiprocessing
import os
from argparse import Namespace
from collections import defaultdict
from os import path

import click
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchnlp.download import download_file_maybe_extract
from tqdm import tqdm

SEPP_NLG_URL = "https://unbabel-experimental-data-sets.s3-eu-west-1.amazonaws.com/video-pt2020/sepp_nlg_2021_train_dev_data.zip"

LABEL_ENCODER_UNBABEL = {
    "0": 0,
    "(": 1,
    ")": 2,
    ":": 3,
    ".": 4,
    ",": 5,
    "/": 6,
    "-": 7,
    "%": 8,
    "&": 9,
    "'": 10,
    ";": 11,
    "!": 12,
    "?": 13,
    "1": 14,
    "+": 15,
    "[": 16,
    "]": 17,
    '"': 18,
    "$": 19,
    "*": 20,
    "=": 21,
    "_": 22,
    "-.": 23,
    "@": 24,
    "\\": 25,
    ">": 26,
    "./": 27,
    "`": 28,
    "'(": 29,
    "<": 30,
    "{": 31,
    "}": 32,
}

LABEL_ENCODER_PAUZEE = {"0": 0, "1": 1, "2": 2, "3": 3, }

MODEL_INPUTS = [
    "input_ids",
    "word_pointer",
    "attention_mask",
    # "token_type_ids",
    "binary_label",
    "punct_label",
]

ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    # "additional_special_tokens": ["<en>", "<de>", "<it>", "<fr>"],
}


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule.
    :param hparams: Namespace with data specific arguments.
    :param tokenizer: Model Tokenizer.
    """

    def __init__(self, hparams: dict, tokenizer, dataset_to_analyse):
        super().__init__()
        self.hparams.update(hparams)
        self.tokenizer = tokenizer
        self.language_pairs = {"fr": 0,}
        self.dataset_to_analyse = dataset_to_analyse

    def preprocess_file_unbabel(self, filename, language, testing=False):
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "word_pointer": [],
        }
        if not testing:
            model_inputs["binary_label"] = []
            model_inputs["punct_label"] = []

        model_inputs["input_ids"].append(
            [
                self.tokenizer.bos_token_id,
            ]
        )
        model_inputs["word_pointer"].append([])
        if not testing:
            model_inputs["binary_label"].append([])
            model_inputs["punct_label"].append([])

        for i, line in enumerate(open(filename).readlines()):
            line = line.strip().split("\t")
            word = line[0] if i == 0 else " " + line[0]
            subwords = self.tokenizer(word, add_special_tokens=False)["input_ids"]

            if (len(model_inputs["input_ids"][-1]) + len(subwords)) < self.tokenizer.model_max_length - 1:
                model_inputs["word_pointer"][-1].append(len(model_inputs["input_ids"][-1]))
                model_inputs["input_ids"][-1] += subwords
                if not testing:
                    model_inputs["binary_label"][-1].append(int(line[1]))
                    model_inputs["punct_label"][-1].append(LABEL_ENCODER_PAUZEE[line[2]])

            else:
                model_inputs["input_ids"][-1].append(self.tokenizer.eos_token_id)
                model_inputs["input_ids"].append([self.tokenizer.bos_token_id,])
                model_inputs["word_pointer"].append([])
                if not testing:
                    model_inputs["binary_label"].append([])
                    model_inputs["punct_label"].append([])

                model_inputs["word_pointer"][-1].append(len(model_inputs["input_ids"][-1]))
                model_inputs["input_ids"][-1] += subwords
                if not testing:
                    model_inputs["binary_label"][-1].append(int(line[1]))
                    model_inputs["punct_label"][-1].append(LABEL_ENCODER_PAUZEE[line[2]])

        if len(model_inputs["input_ids"][-1]) != 1:
            model_inputs["input_ids"][-1].append(self.tokenizer.eos_token_id)

        for _input in model_inputs["input_ids"]:
            model_inputs["attention_mask"].append([1 for _ in _input])

        return model_inputs

    def preprocess_file_pauzee(self, filename, language, testing=False):
        model_inputs = {"input_ids": [], "attention_mask": [], "word_pointer": [],}
        if not testing:
            model_inputs["binary_label"] = []
            model_inputs["punct_label"] = []

        model_inputs["input_ids"].append([self.tokenizer.bos_token_id,])
        model_inputs["word_pointer"].append([])
        if not testing:
            model_inputs["binary_label"].append([])
            model_inputs["punct_label"].append([])

        for i, line in enumerate(open(filename).readlines()[1:]):
            line = line.strip().split(",") # "\t"
            word = line[0] if i == 0 else " " + line[0]
            subwords = self.tokenizer(word, add_special_tokens=False)["input_ids"]

            if (len(model_inputs["input_ids"][-1]) + len(subwords)) < self.tokenizer.model_max_length - 1:
                model_inputs["word_pointer"][-1].append(len(model_inputs["input_ids"][-1]))
                model_inputs["input_ids"][-1] += subwords
                if not testing:
                    model_inputs["binary_label"][-1].append(int(line[1]))
                    model_inputs["punct_label"][-1].append(LABEL_ENCODER_PAUZEE[line[2]])

            else:
                model_inputs["input_ids"][-1].append(self.tokenizer.eos_token_id)
                model_inputs["input_ids"].append([self.tokenizer.bos_token_id,])
                model_inputs["word_pointer"].append([])
                if not testing:
                    model_inputs["binary_label"].append([])
                    model_inputs["punct_label"].append([])

                model_inputs["word_pointer"][-1].append(len(model_inputs["input_ids"][-1]))
                model_inputs["input_ids"][-1] += subwords
                if not testing:
                    model_inputs["binary_label"][-1].append(int(line[1]))
                    model_inputs["punct_label"][-1].append(LABEL_ENCODER_PAUZEE[line[2]])

        if len(model_inputs["input_ids"][-1]) != 1:
            model_inputs["input_ids"][-1].append(self.tokenizer.eos_token_id)

        for _input in model_inputs["input_ids"]:
            model_inputs["attention_mask"].append([1 for _ in _input])

        return model_inputs

    def prepare_data(self):
        dataset = self.dataset_to_analyse
        if not path.isdir(dataset):
            click.secho(f"{dataset} not found.")

        dataset_path = dataset + "/"
        dataset_hash = (int(hashlib.sha256(dataset_path.encode("utf-8")).hexdigest(), 16) % 10 ** 8)
        # To avoid using cache for different models
        # split(/) for microsoft/DialoGPT-small
        pretrained_model = (
            self.hparams.pretrained_model.split("/")[1]
            if "/" in self.hparams.pretrained_model
            else self.hparams.pretrained_model
        )
        dataset_cache = (dataset_path + ".dataset_" + str(dataset_hash) + pretrained_model)

        click.secho(f"Preparing {dataset_cache} data:", fg="red")
        if os.path.isfile(dataset_cache):
            click.secho(f"Loading datasets from cache: {dataset_cache}.")
            tensor_datasets = torch.load(dataset_cache)
        else:
            click.secho(f"Preprocessing: {dataset_cache}.")
            datasets = {"train": defaultdict(list), "dev": defaultdict(list)}
            for lp in self.language_pairs.keys():
                for dataset_name in datasets.keys():
                    click.secho(f"Preparing {dataset_name} data:", fg="yellow")
                    for file in tqdm(
                            os.listdir(dataset_path + "/" + dataset_name + "/"),
                            desc=f"Preparing {lp} data...",
                    ):
                        if file.endswith(".csv"):
                            data = self.preprocess_file_pauzee(dataset_path + "/" + dataset_name + "/" + file, lp)
                            for model_input in data.keys():
                                if model_input in datasets[dataset_name]:
                                    datasets[dataset_name][model_input] += data[model_input]
                                else:
                                    datasets[dataset_name][model_input] = data[model_input]

            click.secho("Padding inputs and building tensors.", fg="yellow")
            tensor_datasets = {"train": [], "dev": []}
            for dataset_name, dataset in datasets.items():
                dataset = self.pad_dataset(dataset, padding=self.tokenizer.pad_token_id)
                for input_name in MODEL_INPUTS:
                    tensor = torch.tensor(dataset[input_name])
                    tensor_datasets[dataset_name].append(tensor)

            tensor_datasets["train"] = TensorDataset(*tensor_datasets["train"])
            tensor_datasets["dev"] = TensorDataset(*tensor_datasets["dev"])
            torch.save(tensor_datasets, dataset_cache)

        self.train_dataset = tensor_datasets["train"]
        self.valid_dataset = tensor_datasets["dev"]
        click.secho(
            "Train dataset (Batch, Candidates, Seq length): {}".format(
                self.train_dataset.tensors[0].shape
            ),
            fg="yellow",
        )
        click.secho("Dev dataset (Batch, Candidates, Seq length): {}".format(self.valid_dataset.tensors[0].shape),
            fg="yellow",
        )

    def pad_dataset(self, dataset: dict, padding: int = 0):
        for input_name in dataset.keys():
            max_l = (
                self.tokenizer.model_max_length
                if "input_ids" in input_name
                else max(len(x) for x in dataset[input_name])
            )
            if input_name == "attention_mask":
                dataset[input_name] = [
                    x + [0] * (self.tokenizer.model_max_length - len(x))
                    for x in dataset[input_name]
                ]
            else:
                dataset[input_name] = [
                    x + [-100 if "label" in input_name else padding] * (max_l - len(x))
                    for x in dataset[input_name]
                ]
        return dataset

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
        )
