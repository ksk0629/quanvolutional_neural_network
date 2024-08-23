import torch


class PlainDataset(torch.utils.data.Dataset):
    """PlainDataset class"""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        """Initialise this dataset.

        :param torch.Tensor x: data
        :param torch.Tensor y: labels
        """
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
