import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm


class Trainer:
    """Trainer class."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        epochs: int,
        save_steps: int,
    ):
        """Initialise this trainer.

        :param nn.Module model: model to train
        :param torch.utils.data.DataLoader train_loader: train data loader
        :param torch.utils.data.DataLoader test_loader: test data loader
        :param int epochs: number of epochs
        :param int save_steps: number of steps to save
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.save_steps = save_steps
        self.current_epoch = 0

        self.criterion = nn.NLLLoss()
        self.optimiser = optim.Adam(self.model.parameters())

        self.train_loss_history = []
        self.test_loss_history = []

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    def calc_loss(self, data: torch.Tensor, label: torch.Tensor) -> nn._Loss:
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
            for data, label in tepoch:
                # Set the description.
                tepoch.set_description(f"Epoch {self.current_epoch} (train)")

                # Transfer the data and label to the device.
                data, label = data.to(self.device), label.to(self.device)

                # Update the parameters.
                loss_value = self.update(data=data, label=label)
                train_loss += loss_value

                # Set the current loss value.
                tepoch.set_postfix(loss=loss_value)

                # Save the parameters according to self.save_steps.
                if self.current_epoch % self.save_steps == 0:
                    output_path = f"model_{self.current_epoch}"
                    torch.save(self.model.state_dict(), output_path)

        # Store the loss value.
        self.train_loss_history.append(train_loss / len(self.train_loader))

    def eval(self):
        """Evaluate the model."""
        self.model.eval()

        test_loss = 0
        with tqdm(self.test_loader) as tepoch:
            with torch.no_grad():  # without calculating the gradients.
                for data, label in tepoch:
                    # Set the description.
                    tepoch.set_description(f"Epoch {self.current_epoch} (test)")

                    # Transfer the data and label to the device.
                    data, label = data.to(self.device), label.to(self.device)
                    # Calculate the loss.
                    loss_value = self.calc_loss(data=data, label=label)
                    test_loss += loss_value

                    # Set the current loss value.
                    tepoch.set_postfix(loss=loss_value)
        # Store the loss value.
        self.test_loss_history.append(test_loss / len(self.test_loader))

    def train_and_test_one_epoch(self):
        """Train and evaluate the model only once."""
        self.train()
        self.eval()

    def train_and_test(self):
        """Train and evaluate the model self.epochs times."""
        for current_epoch in tqdm(range(1, self.epochs + 1)):
            self.current_epoch = current_epoch
            self.train_and_test_one_epoch()
