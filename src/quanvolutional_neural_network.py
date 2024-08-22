import torch

from classical_convolutional_neural_network import ClassicalConvolutionalNeuralNetwork
from quanvolutional_layer import QuanvolutionalLayer


class QuanvolutionalNeuralNetwork:
    def __init__(
        self,
        input_dimension: tuple[int, int, int],
        num_classes: int,
        quanv_kernel_size: tuple[int, int],
        quanv_num_filters: int,
        quanv_padding_mode: str | None = "constant",
    ):
        self.classical_cnn = ClassicalConvolutionalNeuralNetwork(
            input_dimension=input_dimension, num_classes=num_classes
        )
        self.quanvolutional_layer = QuanvolutionalLayer(
            kernel_size=quanv_kernel_size,
            num_filters=quanv_num_filters,
            padding_mode=quanv_padding_mode,
        )

    def __call__(self, x: torch.Tensor, shots: int) -> torch.Tensor:
        quanvoluted_x = self.quanvolutional_layer.run_for_batch(
            batch_data=x, shots=shots
        )
        return self.classical_cnn(quanvoluted_x)

    def classify(self, x: torch.Tensor, shots: int) -> torch.Tensor:
        quanvoluted_x = self.quanvolutional_layer.run_for_batch(
            batch_data=x, shots=shots
        )
        return self.classical_cnn.classify(quanvoluted_x)
