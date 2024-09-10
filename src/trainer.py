import datetime
import os

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from utils import fix_seed


class Trainer:
    """Trainer class."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        batch_size: int,
        epochs: int,
        save_steps: int,
        random_seed: int,
        model_output_dir: str | None,
        model_name: str | None,
    ):
        """Initialise this trainer.

        :param nn.Module model: model to train
        :param torch.utils.data.Dataset train_dataset: train dataset
        :param torch.utils.data.Dataset test_dataset: test dataset
        :param int batch_size: batch size
        :param int epochs: number of epochs
        :param int save_steps: number of steps to save
        :param int random_seed: random seed
        :param str | None model_output_dir: path to output directory
        :param str model_name: model_name
        """
        self.random_seed = random_seed
        fix_seed(self.random_seed)

        self.model = model
        self.epochs = epochs
        self.save_steps = save_steps
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.model_name = model_name if model_name is not None else "model"
        self.current_epoch = 0

        # Make the output directry name with date and time information.
        dt_now = datetime.datetime.now()
        year = str(dt_now.year).zfill(4)
        month = str(dt_now.month).zfill(2)
        day = str(dt_now.day).zfill(2)
        hour = str(dt_now.hour).zfill(2)
        minute = str(dt_now.minute).zfill(2)
        second = str(dt_now.second).zfill(2)
        postfix = f"{year}{month}{day}{hour}{minute}{second}"
        self.model_output_dir = f"{model_output_dir}_{postfix}"

        # self.criterion = nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = optim.Adam(self.model.parameters())

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.train_loss_history = []
        self.test_loss_history = []

        self.train_accuracy_history = []
        self.test_accuracy_history = []

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create the output directory if not exsiting.
        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)

    def update(self, data: torch.Tensor, label: torch.Tensor) -> float:
        """Update the parameters of the model.

        :param torch.Tensor data: data for training
        :param torch.Tensor label: label for training
        :return float: loss value
        """
        # Initialise the gradients.
        self.optimiser.zero_grad()
        # Calculate the loss.
        loss = self.calc_loss(data=data, label=label)
        # Perform the backpropagation.
        loss.backward()
        # Update the parameters.
        self.optimiser.step()

        return loss.item()

    def calc_loss(
        self, data: torch.Tensor, label: torch.Tensor
    ) -> torch.nn.modules.loss._Loss:
        """Calculate the loss.

        :param torch.Tensor data: data for calculating loss
        :param torch.Tensor label: data for calculating loss
        :return nn._Loss: loss
        """
        # Classify the data.
        output = self.model(data)
        # Calculate the loss value and accumulate it.
        loss = self.criterion(output, label)

        return loss

    def train(self):
        """Train the model."""
        self.model.train()

        train_loss = 0
        with tqdm(self.train_loader) as tepoch:
            # Initialise the count of correctly predicted data.
            total_correct = 0
            total = 0

            for data, label in tepoch:
                # Set the description.
                tepoch.set_description(f"Epoch {self.current_epoch} (train)")

                # Transfer the data and label to the device.
                data, label = data.to(self.device), label.to(self.device)

                # Update the parameters.
                loss_value = self.update(data=data, label=label)
                train_loss += loss_value

                # Get the number of correctly predicted ones.
                predicted_label = self.model.classify(data)
                num_correct = (predicted_label == label).sum().item()
                total_correct += num_correct
                total += len(label)

                # Set the current loss and accuracy.
                batch_accuracy = num_correct / len(label)
                tepoch.set_postfix(
                    {"Loss_train": loss_value, "Accuracy_train": batch_accuracy}
                )

                # Save the parameters according to self.save_steps.
                if self.current_epoch % self.save_steps == 0:
                    filename = f"{self.model_name}_{self.current_epoch}.pth"
                    output_path = os.path.join(self.model_output_dir, filename)
                    torch.save(self.model.state_dict(), output_path)

        # Store the loss value.
        average_train_loss = train_loss / len(self.train_loader)
        self.train_loss_history.append(average_train_loss)
        mlflow.log_metric(f"train_loss", average_train_loss, step=self.current_epoch)

        # Store the accuracy.
        accuracy = total_correct / total
        self.train_accuracy_history.append(accuracy)
        mlflow.log_metric(f"train_accuracy", accuracy, step=self.current_epoch)

    def eval(self):
        """Evaluate the model."""
        self.model.eval()

        test_loss = 0
        with tqdm(self.test_loader) as tepoch:
            with torch.no_grad():  # without calculating the gradients.
                # Initialise the count of correctly predicted data.
                total_correct = 0
                total = 0

                for data, label in tepoch:
                    # Set the description.
                    tepoch.set_description(f"Epoch {self.current_epoch} (test)")

                    # Transfer the data and label to the device.
                    data, label = data.to(self.device), label.to(self.device)
                    # Calculate the loss.
                    loss_value = self.calc_loss(data=data, label=label)
                    test_loss += loss_value

                    # Get the number of correctly predicted ones.
                    predicted_label = self.model.classify(data)
                    num_correct = (predicted_label == label).sum().item()
                    total_correct += num_correct
                    total += len(label)

                    # Set the current loss and accuracy.
                    batch_accuracy = num_correct / len(label)
                    tepoch.set_postfix(
                        {
                            "Loss_test": loss_value,
                            "Accuracy_test": batch_accuracy,
                        }
                    )
        # Store the loss value.
        average_test_loss = test_loss / len(self.test_loader)
        self.test_loss_history.append(average_test_loss)
        mlflow.log_metric(f"test_loss", average_test_loss, step=self.current_epoch)

        # Store the accuracy.
        accuracy = total_correct / total
        self.test_accuracy_history.append(accuracy)
        mlflow.log_metric(f"test_accuracy", accuracy, step=self.current_epoch)

    def train_and_test_one_epoch(self):
        """Train and evaluate the model only once."""
        self.train()
        self.eval()

    def train_and_test(self):
        """Train and evaluate the model self.epochs times."""
        for current_epoch in range(1, self.epochs + 1):
            self.current_epoch = current_epoch
            self.train_and_test_one_epoch()

        filename = f"{self.model_name}_final_{self.epochs}.pth"
        output_path = os.path.join(self.model_output_dir, filename)
        torch.save(self.model.state_dict(), output_path)

        mlflow.pytorch.log_model(self.model, "classical_cnn")
