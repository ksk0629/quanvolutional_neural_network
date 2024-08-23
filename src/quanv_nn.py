import torch

from classical_cnn import ClassicalCNN
from quanv_layer import QuanvLayer


class QuanvNN:
    def __init__(
        self,
        in_dim: tuple[int, int, int],
        num_classes: int,
        quanv_kernel_size: tuple[int, int],
        quanv_num_filters: int,
        quanv_padding_mode: str | None = "constant",
    ):
        """Initialise this QNN.

        :param tuple[int, int, int] in_dim: input data dimension formed as [channels, height, width]
        :param int num_classes: number of clssses to classify
        :param tuple[int, int] quanv_kernel_size: size of kernel for quanvolutional layer
        :param int quanv_num_filters: number of quanvolutional filters
        :param str | None quanv_padding_mode: padding mode (see the document of torch.nn.functional.pad), defaults to "constant"
        """
        self.quanv_layer = QuanvLayer(
            kernel_size=quanv_kernel_size,
            num_filters=quanv_num_filters,
            padding_mode=quanv_padding_mode,
        )
        new_in_dim = (quanv_num_filters, in_dim[1], in_dim[2])
        self.classical_cnn = ClassicalCNN(in_dim=new_in_dim, num_classes=num_classes)

    def __call__(self, x: torch.Tensor, shots: int) -> torch.Tensor:
        """forward a batch data.

        :param torch.Tensor x: batch data
        :param int shots: number of shots
        :return torch.Tensor: processed data
        """
        quanvoluted_x = self.quanv_layer.run_for_batch(batch_data=x, shots=shots)
        return self.classical_cnn(quanvoluted_x)

    def classify(self, x: torch.Tensor, shots: int) -> torch.Tensor:
        """Classify a batch data.

        :param torch.Tensor x: batch data
        :param int shots: number of shots
        :return torch.Tensor: result of classification
        """
        quanvoluted_x = self.quanv_layer.run_for_batch(batch_data=x, shots=shots)
        return self.classical_cnn.classify(quanvoluted_x)
