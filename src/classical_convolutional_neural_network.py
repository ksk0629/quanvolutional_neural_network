import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalCNN(nn.Module):
    """Classical CNN class."""

    def __init__(self, in_dim: tuple[int, int, int], num_classes: int):
        """Initialise this CNN.

        :param tuple[int, int, int] in_dim: input data dimension formed as [channels, height, width]
        :param int num_classes: number of clssses to classify
        """
        super().__init__()
        self.kernel_size = 5
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.pool_size = 2
        self.num_classes = num_classes
        self.relu = F.relu
        self.softmax = F.softmax

        self.num_filter1 = 50
        self.conv1 = nn.Conv2d(
            in_channels=in_dim[0],
            out_channels=self.num_filter1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self.conv1_output_shape = self.calc_output_shape(
            in_dim[1],
            in_dim[2],
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
        self.pool1_output_shape = self.calc_output_shape(
            self.conv1_output_shape[0],
            self.conv1_output_shape[1],
            self.pool_size,
            self.pool_size,
            self.padding,
            self.dilation,
        )

        self.num_filter2 = 64
        self.conv2 = nn.Conv2d(
            in_channels=self.num_filter1,
            out_channels=self.num_filter2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self.conv2_output_shape = self.calc_output_shape(
            self.pool1_output_shape[0],
            self.pool1_output_shape[1],
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
        self.pool2_output_shape = self.calc_output_shape(
            self.conv2_output_shape[0],
            self.conv2_output_shape[1],
            self.pool_size,
            self.pool_size,
            self.padding,
            self.dilation,
        )

        self.fc1_input_size = (
            self.num_filter2 * self.pool2_output_shape[0] * self.pool2_output_shape[1]
        )
        self.fc1_output_size = 1024
        self.fc1 = nn.Linear(
            in_features=self.fc1_input_size, out_features=self.fc1_output_size
        )

        self.dropout = nn.Dropout(p=0.4)

        self.fc2 = nn.Linear(
            in_features=self.fc1_output_size, out_features=self.num_classes
        )

    def calc_output_shape(
        self,
        in_height: int,
        in_width: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ) -> tuple[int, int]:
        """Calculate an output shape.

        :param int in_height: input height
        :param int in_width: input width
        :param int kernel_size: kernel size
        :param int stride: stride
        :return tuple[int, int]: output shape
        """
        output_height = math.floor(
            (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )
        output_width = math.floor(
            (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )
        return (output_height, output_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward data.

        :param torch.Tensor x: input data
        :return torch.Tensor: processed data
        """
        # The first convolutional and pooling layers result.
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # The second convolutional and pooling layers result.
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Transform the output shape to input fully connected layer.
        x = x.view(x.size()[0], -1)

        x = self.fc1(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return self.softmax(x, dim=1)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Classify data.

        :param torch.Tensor x: _description_
        :return torch.Tensor: _description_
        """
        probabilities = self.forward(x)
        return torch.argmax(probabilities, dim=1)
