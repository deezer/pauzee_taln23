# -*- coding: utf-8 -*-
r"""
Command Line Interface
=======================
   Commands:
   - train: for Training a new model.
   - interact: Model interactive mode where we can "talk" with a trained model.
   - test: Tests the model ability to rank candidate answers and generate text.
"""
import math
import multiprocessing
import os
from functools import partial
import click
import optuna
import pandas as pd
import torch
import yaml
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.data_module import MODEL_INPUTS, DataModule
from models.punct_predictor import PunctuationPredictor
from trainer import TrainerConfig, build_trainer


@click.group()
def cli():
    pass


@cli.command(name="train")
@click.option("--config", "-f", type=click.Path(exists=True), required=True,
              help="Path to the configure YAML file",)
@click.option("--dataset", type=click.Path(exists=True),
              default="break_detection/data/orfeo-synpaflex",
              help="Path to the folder containing the dataset.",
              )
def train(config: str, dataset: str) -> None:
    yaml_file = yaml.load(open(config).read(), Loader=yaml.FullLoader)
    # Build Trainer
    train_configs = TrainerConfig(yaml_file)
    seed_everything(train_configs.seed)
    trainer = build_trainer(train_configs.namespace())

    # Print Trainer parameters into terminal
    result = "Hyperparameters:\n"
    for k, v in train_configs.namespace().__dict__.items():
        result += "{0:30}| {1}\n".format(k, v)
    click.secho(f"{result}", fg="blue", nl=False)

    model_config = PunctuationPredictor.ModelConfig(yaml_file)
    # Print Model parameters into terminal
    for k, v in model_config.namespace().__dict__.items():
        result += "{0:30}| {1}\n".format(k, v)
    click.secho(f"{result}", fg="cyan")
    model = PunctuationPredictor(model_config.namespace())
    data = DataModule(model.hparams, model.tokenizer, dataset)
    trainer.fit(model, data)


@cli.command(name="test")
@click.option("--model",
              type=click.Path(exists=True),
              required=True,
              help="Folder containing the config files and model checkpoints.",
              )
@click.option("--language",
              type=click.Choice(["en", "de", "fr", "it"], case_sensitive=False),
              default="fr",
              help="Language pair",
              required=False,
              )
@click.option("--test/--dev",
              default=True,
              help="Flag that either runs devset or testset.",
              show_default=True,
              )
@click.option("--dataset",
              type=click.Path(exists=True),
              default="break_detection/data/orfeo-synpaflex/",
              help="Path to the folder containing the dataset.",
              )
@click.option("--prediction_dir",
              type=click.Path(exists=True),
              default="break_detection/results/orfeo-synpaflex_pauzee/",
              help="Folder used to save predictions.",
              )
@click.option("--prompt",
              default="",
              help="text to test",
              )
@click.option("--batch_size", default=32, help="Batch size used during inference.", type=int,)
def predict(
        model: str,
        language: str,
        test: bool,
        dataset: str,
        prediction_dir: str,
        prompt: str,
        batch_size: int, ) -> None:

    """Testing function where a trained model is tested in its ability to rank candidate
    answers and produce replies.
    """

    # Fix paths
    model = model if model.endswith("/") else model + "/"
    dataset = dataset if dataset.endswith("/") else dataset + "/"
    test_folder = dataset + ("test/" if test else "dev/")

    if prompt != "":
        text = prompt.split(" ")
        text = ",0,0\n".join(text)
        text = "word,break,break_size\n" + text + ",0,0\n"

        os.makedirs("pauzee_tmp_sentence/", exist_ok=True)
        test_folder = "pauzee_tmp_sentence/"
        with open(test_folder + 'pauzee_tmp_sentence2test.csv', 'w') as f:
            f.write(text)
        dataset = test_folder
    else:
        prediction_dir = (prediction_dir if prediction_dir.endswith("/") else prediction_dir + "/")
        output_folder = prediction_dir + ("test/" if test else "dev/")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    click.secho(f"Loading model from folder: {model}", fg="yellow")
    model = PunctuationPredictor.from_experiment(model).to("cuda")
    data_module = DataModule(model.hparams, model.tokenizer, dataset)

    for file in tqdm(os.listdir(test_folder), desc="Processing {} data...".format(test_folder),):
        if file.endswith(".csv"):
            model_inputs = data_module.preprocess_file_pauzee(test_folder + file, language)
            model_inputs = data_module.pad_dataset(model_inputs, padding=model.tokenizer.pad_token_id)
            file_data = []
            for input_name in MODEL_INPUTS:
                tensor = torch.tensor(model_inputs[input_name])
                file_data.append(tensor)

            dataloader = DataLoader(TensorDataset(*file_data), batch_size=batch_size, shuffle=False,
                                    num_workers=multiprocessing.cpu_count(), pin_memory=True,)
            binary_labels, punct_labels = [], []
            for batch in dataloader:
                bin_y_hat, punct_y_hat = model.predict(batch)
                binary_labels += bin_y_hat
                punct_labels += punct_y_hat
            words = [line.strip().split(",")[0] for line in open(test_folder + file).readlines()[1:]]
            output = pd.DataFrame({"word": words, "break": binary_labels, "break_size": punct_labels})
            if prompt == "":
                click.secho(f"writing results here : {output_folder + file}", fg="yellow")
                output.to_csv(output_folder + file, sep="\t", header=None, index=False)
            else:
                print(output)

    if prompt != "":
        os.remove(test_folder + '/pauzee_tmp_sentence2test.csv')
        os.rmdir(test_folder)


if __name__ == "__main__":
    cli()
