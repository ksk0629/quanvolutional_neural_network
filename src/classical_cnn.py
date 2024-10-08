import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalCNN(nn.Module):
    """Classical CNN class."""

    def __init__(
        self,
        in_dim: tuple[int, int, int],
        num_classes: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        pool_size: int = 1,
        num_conv1_filter: int = 50,
        num_conv2_filter: int = 64,
    ):
        """Initialise this CNN.

        :param tuple[int, int, int] in_dim: input data dimension formed as [channels, height, width]
        :param int num_classes: number of classes to classify
        :param int kernel_size: kernel size, defaults to 5
        :param int stride: stride size, defaults to 1
        :param int padding: padding size, defaults to 0
        :param int dilation: dilation size, defaults to 1
        :param int pool_size: pooling size, defaults to 1
        :param int num_conv1_filter: number of conv1 filters, defaults to 50
        :param int num_conv2_filter: number of conv2 filters, defaults to 64
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool_size = pool_size
        self.num_classes = num_classes

        # Set the first convolutional layer.
        self.num_conv1_filter = num_conv1_filter
        self.conv1 = nn.Conv2d(
            in_channels=in_dim[0],
            out_channels=self.num_conv1_filter,
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

        # Set the first pooling layer.
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
        self.pool1_output_shape = self.calc_output_shape(
            self.conv1_output_shape[0],
            self.conv1_output_shape[1],
            self.pool_size,
            self.pool_size,
            self.padding,
            self.dilation,
        )

        # Set the second convolutional layer.
        self.num_conv2_filter = num_conv2_filter
        self.conv2 = nn.Conv2d(
            in_channels=self.num_conv1_filter,
            out_channels=self.num_conv2_filter,
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

        # Set the second pooling layer.
        self.pool2 = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
        self.pool2_output_shape = self.calc_output_shape(
            self.conv2_output_shape[0],
            self.conv2_output_shape[1],
            self.pool_size,
            self.pool_size,
            self.padding,
            self.dilation,
        )

        # Set the first fully connected layer.
        self.fc1_input_size = (
            self.num_conv2_filter
            * self.pool2_output_shape[0]
            * self.pool2_output_shape[1]
        )
        self.fc1_output_size = 1024
        self.fc1 = nn.Linear(
            in_features=self.fc1_input_size, out_features=self.fc1_output_size
        )

        # Set the dropout layer.
        self.dropout = nn.Dropout(p=0.4)

        # Set the second fully connected layer.
        self.fc2 = nn.Linear(
            in_features=self.fc1_output_size, out_features=self.num_classes
        )

    def activate(self, x: torch.Tensor) -> callable:
        """Return the return value of the specified activation function, which is relu now.

        :param torch.Tensor x: input data
        :return callable: activation function
        """
        return F.relu(x)

    def softmax(self, x: torch.Tensor) -> F.softmax:
        """Return the return value of the softmax function.

        :param torch.Tensor x: input data
        :return F.softmax: softmax function
        """
        return F.softmax(x, dim=1)

    def calc_output_shape(
        self,
        in_height: int,
        in_width: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ) -> tuple[int, int]:
        """Calculate an output shape of convolutional or pooling layers.
        Referred to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.

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
        x = self.conv1(x)
        x = self.activate(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.activate(x)
        x = self.pool2(x)

        # Transform the output shape to input the fully connected layer.
        x = x.view(x.size()[0], -1)

        x = self.fc1(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return self.softmax(x)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Classify data.

        :param torch.Tensor x: input data
        :return torch.Tensor: predicted label
        """
        probabilities = self.forward(x)
        return torch.argmax(probabilities, dim=1)
