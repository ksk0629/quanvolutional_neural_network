import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm


class Trainer:
    """Trainer class."""
    def __init__(self,
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 epochs: int,
                 save_steps: int):
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
    
    def train(self):
        self.model.train()
        
        train_loss = 0
        with tqdm(self.train_loader) as tepoch:
            for data, label in tepoch:
                # Set the description.
                tepoch.set_description(f"Epoch {self.current_epoch} (train)")
                
                # Transfer the data and label to the device.
                data, label = data.to(self.device), label.to(self.device)
                # Initialise the gradients.
                self.optimiser.zero_grad()
                # Classify the data.
                output = self.model(data)
                # Calculate the loss value and accumulate it.
                loss = self.criterion(output, label)
                train_loss += loss.item()
                # Perform the backpropagation.
                loss.backward()
                # Update the parameters.
                self.optimiser.step()
                
                # Set the current loss value.
                tepoch.set_postfix(loss=loss.item())
        # Store the loss value.
        self.train_loss_history.append(train_loss / len(self.train_loader))
    
    def test(self):
        self.model.eval()
        
        test_loss = 0
        with tqdm(self.test_loader) as tepoch:
            with torch.no_grad():  # without calculating the gradients.
                for data, label in tepoch:
                    # Set the description.
                    tepoch.set_description(f"Epoch {self.current_epoch} (test)")
                    
                    # Transfer the data and label to the device.
                    data, label = data.to(self.device), label.to(self.device)
                    # Classify the data.
                    output = self.model(data)
                    # Calculate the loss value and accumulate it.
                    loss = self.criterion(output, label)
                    test_loss += loss.item()
                    
                    # Set the current loss value.
                    tepoch.set_postfix(loss=loss.item())
        # Store the loss value.
        self.test_loss_history.append(test_loss / len(self.test_loader))
    
    def train_and_test_one_epoch(self):
        self.train()
        self.test()
    
    def train_and_test(self):
        for current_epoch in tqdm(range(1, self.epochs+1)):
            self.current_epoch = current_epoch
            self.train_and_test_one_epoch()