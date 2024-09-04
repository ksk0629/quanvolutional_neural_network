import pytest
import torch

from src.plain_dataset import PlainDataset


class TestPlainDataset:
    """Test class for src.plain_dataset.PlainDataset class."""

    @classmethod
    def setup_class(self):
        """Setup this test class."""
        self.num_data = 5
        self.normal_x = torch.rand(self.num_data, 5)
        self.normal_y = torch.rand(self.num_data, 1)

    @pytest.mark.dataset
    def test_use_normal_args(self):
        """Normal test with normal arguments;
        No error happens.
        """
        plain_dataset = PlainDataset(self.normal_x, self.normal_y)

    @pytest.mark.dataset
    def test_use_normal_args_length(self):
        """Normal test with normal arguments;
        The length of the dataset is the same as self.num_data.
        """
        plain_dataset = PlainDataset(self.normal_x, self.normal_y)
        assert len(plain_dataset) == self.num_data

    @pytest.mark.dataset
    def test_use_normal_args_items(self):
        """Normal test with normal arguments;
        The length of the dataset is the same as self.num_data.
        """
        plain_dataset = PlainDataset(self.normal_x, self.normal_y)
        for index in range(self.num_data):
            x, y = plain_dataset[index]
            assert torch.eq(x, self.normal_x[index]).all()
            assert torch.eq(y, self.normal_y[index]).all()
