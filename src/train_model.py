import sys
import yaml

import mlflow
import torch

sys.path.append("./../src")
from classical_cnn import ClassicalCNN
from one_sum_decoder import OneSumDecoder
from quanv_nn import QuanvNN
from quanv_nn_trainer import QuanvNNTrainer
from trainer import Trainer
from z_basis_encoder import ZBasisEncoder


def prepare_classical_cnn(
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    # Arguments for a model
    in_dim: tuple[int, int, int],
    num_classes: int,
    # Arguments for training
    batch_size: int,
    epochs: int,
    save_steps: int,
    random_seed: int,
    model_output_dir: str | None,
    model_name: str | None,
) -> Trainer:
    """Prepare the trainer for ClassicalCNN.

    :param torch.utils.data.Dataset train_dataset: train dataset
    :param torch.utils.data.Dataset test_dataset: test dataset
    :param tuple[int, int, int] in_dim: input data dimension formed as [channels, height, width]
    :param int num_classes: number of classes to classify
    :param int batch_size: batch size
    :param int epochs: number of epochs
    :param int save_steps: number of steps to save
    :param int random_seed: random seed
    :param str | None model_output_dir: path to output directory
    :param str model_name: model_name
    :return Trainer: trainer for ClassicalCNN
    """
    classical_cnn = ClassicalCNN(in_dim=in_dim, num_classes=num_classes)
    trainer = Trainer(
        model=classical_cnn,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        epochs=epochs,
        save_steps=save_steps,
        random_seed=random_seed,
        model_output_dir=model_output_dir,
        model_name=model_name,
    )
    return trainer


def prepare_quanv_nn_trainer(
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    # Arguments for a model
    in_dim: tuple[int, int, int],
    num_classes: int,
    quanv_kernel_size: tuple[int, int],
    quanv_num_filters: int,
    quanv_padding_mode: str | None,
    # Arguments for training
    epochs: int,
    batch_size: int,
    save_steps: int,
    random_seed: int,
    shots: int,
    model_output_dir: str,
    model_name: str,
    processed_data_filename: str | None,
    is_lookup_mode: bool,
) -> QuanvNNTrainer:
    """Prepare the trainer for QuanvNN.

    :param torch.utils.data.Dataset train_dataset: dataset for training
    :param torch.utils.data.Dataset test_dataset: dataset for test
    :param tuple[int, int, int] in_dim: input data dimension formed as [channels, height, width]
    :param int num_classes: number of classes to classify
    :param tuple[int, int] quanv_kernel_size: size of kernel for quanvolutional layer
    :param int quanv_num_filters: number of quanvolutional filters
    :param str | None quanv_padding_mode: padding mode (see the document of torch.nn.functional.pad)
    :param int epochs: number of epochs
    :param int batch_size: batch size
    :param int save_steps: number of steps to save
    :param int random_seed: random seed
    :param int shots: number of shots
    :param str model_output_dir: path to model output directory
    :param str model_name: model_name
    :param str | None processed_data_filename: processed data filename to output
    :param bool is_lookup_mode: if it is look-up mode
    :return QuanvNNTrainer: trainer for QuanvNN
    """
    quanv_nn = QuanvNN(
        in_dim=in_dim,
        num_classes=num_classes,
        quanv_kernel_size=quanv_kernel_size,
        quanv_num_filters=quanv_num_filters,
        quanv_encoder=ZBasisEncoder(),
        quanv_decoder=OneSumDecoder(),
        quanv_padding_mode=quanv_padding_mode,
    )
    quanv_nn_trainer = QuanvNNTrainer(
        qnn=quanv_nn,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        epochs=epochs,
        batch_size=batch_size,
        save_steps=save_steps,
        random_seed=random_seed,
        shots=shots,
        model_output_dir=model_output_dir,
        model_name=model_name,
        processed_data_filename=processed_data_filename,
        is_lookup_mode=is_lookup_mode,
    )
    return quanv_nn_trainer


def train(
    config_yaml_path: str,
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    num_classes: int,
):
    """Train a model according to a given config.

    :param str config_yaml_path: path to config file
    :param torch.utils.data.Dataset train_dataset: train dataset
    :param torch.utils.data.Dataset test_dataset: test dataset
    :param int num_classes: number of classes
    """
    # Read the given config file.
    with open(config_yaml_path, "r") as config_yaml:
        config = yaml.safe_load(config_yaml)
    config_mlflow = config["mlflow"]
    config_train = config["train"]
    if "model" in config:
        # Get the model config if existed.
        config_model = config["model"]
    else:
        # Create model config as an empty dictionary.
        config_model = dict()

    # Get in_dim and num_classes from the dataset.
    sample_data = train_dataset[0][0]
    config_model["in_dim"] = sample_data.shape
    config_model["num_classes"] = num_classes

    # Start mlflow.
    mlflow.set_experiment(config_mlflow["experiment_name"])
    with mlflow.start_run(run_name=config_mlflow["run_name"]):
        # Log parameters.
        mlflow.log_params(config_model)
        mlflow.log_params(config_train)

        # Get the appropriate trainer.
        if "quanv_kernel_size" in config_model:
            # Convert list to tuple as Yaml can't handle tuple but the code assumes tuple.
            config_model["quanv_kernel_size"] = tuple(config_model["quanv_kernel_size"])
            trainer = prepare_quanv_nn_trainer(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                **config_model,
                **config_train
            )
        else:
            trainer = prepare_classical_cnn(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                **config_model,
                **config_train
            )

        # Train and test.
        trainer.train_and_test()

        mlflow.log_artifact(config_yaml_path)
