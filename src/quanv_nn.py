import json
import os

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
        is_lookup_mode: bool = True,
    ):
        """Initialise this QNN.

        :param tuple[int, int, int] in_dim: input data dimension formed as [channels, height, width]
        :param int num_classes: number of clssses to classify
        :param tuple[int, int] quanv_kernel_size: size of kernel for quanvolutional layer
        :param int quanv_num_filters: number of quanvolutional filters
        :param str | None quanv_padding_mode: padding mode (see the document of torch.nn.functional.pad), defaults to "constant"
        :param bool is_lookup_mode: if it is look-up mode, defaults to True
        """
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.quanv_kernel_size = quanv_kernel_size
        self.quanv_num_filters = quanv_num_filters
        self.quanv_padding_mode = quanv_padding_mode
        self.is_lookup_mode = is_lookup_mode

        self.quanv_layer = QuanvLayer(
            kernel_size=quanv_kernel_size,
            num_filters=quanv_num_filters,
            padding_mode=quanv_padding_mode,
            is_lookup_mode=is_lookup_mode,
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

    def save(self, output_dir: str, filename_prefix: str):
        """Save the QNN config.

        :param str output_dir: path to output dir
        :param str filename_prefix: prefix of output files
        """
        # Create the output directory.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the classical CNN.
        classical_cnn_output_dir = os.path.join(output_dir, "classical_cnn")
        if not os.path.exists(classical_cnn_output_dir):
            os.makedirs(classical_cnn_output_dir)
        classical_cnn_filename = f"{filename_prefix}_classical_cnn_config.pth"
        classical_cnn_path = os.path.join(
            classical_cnn_output_dir, classical_cnn_filename
        )
        torch.save(self.classical_cnn.state_dict(), classical_cnn_path)

        # Make and save the config.
        config = dict()
        config["in_dim"] = self.in_dim
        config["num_classes"] = self.num_classes
        config["quanv_kernel_size"] = self.quanv_kernel_size
        config["quanv_num_filters"] = self.quanv_num_filters
        config["quanv_padding_mode"] = self.quanv_padding_mode
        config_filename = f"{filename_prefix}_quanv_nn_config.json"
        quanv_output_dir = os.path.join(output_dir, "quanv")
        if not os.path.exists(quanv_output_dir):
            os.makedirs(quanv_output_dir)
        config_path = os.path.join(quanv_output_dir, config_filename)
        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=4)

        # Save each QuanvFilter.
        for index, quanv_filter in enumerate(self.quanv_layer.quanv_filters):
            quanv_filter_filename_prefix = f"{filename_prefix}_{index}"
            quanv_filter.save(
                output_dir=quanv_output_dir,
                filename_prefix=quanv_filter_filename_prefix,
            )
