import os

import torch

from plain_dataset import PlainDataset
from trainer import Trainer
from quanv_nn import QuanvNN
from utils import fix_seed


class QuanvNNTrainer:
    """QuanvNNTrainer class."""

    def __init__(
        self,
        qnn: QuanvNN,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        epochs: int,
        batch_size: int,
        save_steps: int,
        random_seed: int,
        shots: int,
        model_output_dir: str,
        model_name: str,
        processed_data_filename: str | None,
        processed_data_output_dir: str | None = "./../data",
    ):
        """Initialise this trainer.

        :param QuanvNN qnn: quanvolutional neural network
        :param torch.utils.data.Dataset train_dataset: dataset for training
        :param torch.utils.data.Dataset test_dataset: dataset for test
        :param int epochs: number of epochs
        :param int batch_size: batch size
        :param int save_steps: number of steps to save
        :param int random_seed: random seed
        :param int shots: number of shots
        :param str model_output_dir: path to model output directory
        :param str model_name: model_name
        :param str | None processed_data_filename: processed data filename to output
        :param str | None processed_data_output_dir: path to processed data output directory, defaults to "./../data"
        """
        self.random_seed = random_seed
        fix_seed(self.random_seed)

        self.qnn = qnn
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_steps = save_steps
        self.shots = shots
        self.model_output_dir = model_output_dir
        self.model_name = model_name
        self.processed_data_filename = processed_data_filename
        self.processed_data_output_dir = processed_data_output_dir

        # Create the output directory if it is given and does not exist.
        is_processed_data_output_dir_given = self.processed_data_output_dir is not None
        if is_processed_data_output_dir_given:
            if not os.path.exists(self.processed_data_output_dir):
                os.makedirs(self.processed_data_output_dir)

    def preprocess(self, dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
        """Preprocess a given dataset using the QuanvLayer.

        :param torch.utils.data.Dataset dataset: dataset to preprocess
        :return torch.utils.data.Dataset: preprocessed dataset
        """
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset), shuffle=False
        )
        data, labels = next(iter(data_loader))
        processed_data = self.qnn.quanv_layer.run_for_batch(data, shots=self.shots)
        processed_dataset = PlainDataset(processed_data, labels)
        return processed_dataset

    def set_preprocessed_datasets(self):
        """Set preprocessed training and test data loaders."""
        # Check if preprocessed data should be saved or not.
        is_processed_data_output_dir_given = self.processed_data_output_dir is not None
        is_processed_data_filename_given = self.processed_data_output_dir is not None
        processed_data_save_flag = (
            is_processed_data_filename_given and is_processed_data_output_dir_given
        )

        # Processed the train data using QuanvLayer and set it as train_loader.
        self.preprocessed_train_dataset = self.preprocess(self.train_dataset)
        if processed_data_save_flag:
            # Save the data.
            train_output_path = os.path.join(
                self.processed_data_output_dir, "train_" + self.processed_data_filename
            )
            torch.save(self.preprocessed_train_dataset, train_output_path)

        # Processed the test data using QuanvLayer and set it as test_loader.
        self.preprocessed_test_dataset = self.preprocess(self.test_dataset)
        if processed_data_save_flag:
            # Save the data.
            test_output_path = os.path.join(
                self.processed_data_output_dir, "test_" + self.processed_data_filename
            )
            torch.save(self.preprocessed_test_dataset, test_output_path)

    def train_and_test(self):
        """Train and test the QuanvNN."""
        # 1: Make a processed datasets.
        self.set_preprocessed_datasets()

        # 2: Make Trainer instance with the preprocessed datasets.
        self.trainer = Trainer(
            model=self.qnn.classical_cnn,
            train_dataset=self.preprocessed_train_dataset,
            test_dataset=self.preprocessed_test_dataset,
            epochs=self.epochs,
            save_steps=self.save_steps,
            output_dir=self.model_output_dir,
            model_name=self.model_name,
        )

        # 3: Train the classical part of self.qnn.
        self.trainer.train_and_test()
