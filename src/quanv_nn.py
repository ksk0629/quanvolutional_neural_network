import json
import os
import pickle

import torch

from decoders.base_decoder import BaseDecoder
from encoders.base_encoder import BaseEncoder
from classical_cnn import ClassicalCNN
from quanv_layer import QuanvLayer


class QuanvNN:
    """Quanvolutional neural network class."""

    def __init__(
        self,
        in_dim: tuple[int, int, int],
        num_classes: int,
        quanv_kernel_size: tuple[int, int],
        quanv_num_filters: int,
        quanv_encoder: BaseEncoder,
        quanv_decoder: BaseDecoder,
        quanv_padding_mode: str | None = "constant",
        is_lookup_mode: bool = True,
    ):
        """Initialise this QNN.

        :param tuple[int, int, int] in_dim: input data dimension formed as [channels, height, width]
        :param int num_classes: number of classes to classify
        :param tuple[int, int] quanv_kernel_size: size of kernel for quanvolutional layer
        :param int quanv_num_filters: number of quanvolutional filters
        :param BaseEncoder quanv_encoder: encoder for quanvolutional layer
        :param BaseDecoder quanv_decoder: decoder for quanvolutional layer
        :param str | None quanv_padding_mode: padding mode (see the document of torch.nn.functional.pad), defaults to "constant"
        :param bool is_lookup_mode: if it is look-up mode, defaults to True
        """
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.quanv_kernel_size = quanv_kernel_size
        self.quanv_num_filters = quanv_num_filters
        self.quanv_encoder = quanv_encoder
        self.quanv_decoder = quanv_decoder
        self.quanv_padding_mode = quanv_padding_mode
        self.is_lookup_mode = is_lookup_mode

        # Create and store the instance of the QuanvLayer class as a member variable.
        self.quanv_layer = QuanvLayer(
            kernel_size=quanv_kernel_size,
            num_filters=quanv_num_filters,
            encoder=self.quanv_encoder,
            decoder=self.quanv_decoder,
            padding_mode=quanv_padding_mode,
            is_lookup_mode=is_lookup_mode,
        )
        # Create and store the instance of the ClassicalCNN class as a member variable.
        new_in_dim = (quanv_num_filters * in_dim[0], in_dim[1], in_dim[2])
        self.classical_cnn = ClassicalCNN(in_dim=new_in_dim, num_classes=num_classes)

    def __call__(self, x: torch.Tensor, shots: int) -> torch.Tensor:
        """forward a batch data.

        :param torch.Tensor x: batch data
        :param int shots: number of shots
        :return torch.Tensor: processed data
        """
        quanvoluted_x = self.quanv_layer.run(batch_data=x, shots=shots)
        return self.classical_cnn(quanvoluted_x)

    def classify(self, x: torch.Tensor, shots: int) -> torch.Tensor:
        """Classify a batch data.

        :param torch.Tensor x: batch data
        :param int shots: number of shots
        :return torch.Tensor: result of classification
        """
        quanvoluted_x = self.quanv_layer.run(batch_data=x, shots=shots)
        return self.classical_cnn.classify(quanvoluted_x)

    def get_classical_cnn_filename(self, filename_prefix: str) -> str:
        """Get ClassicalCNN filename.

        :param str filename_prefix: prefix of filename
        :return str: ClassicalCNN filename
        """
        return f"{filename_prefix}_classical_cnn_config.pth"

    def get_qnn_config_filename(self, filename_prefix: str) -> str:
        """Get QNN config filename.

        :param str filename_prefix: prefix of filename
        :return str: QNN config filename
        """
        return f"{filename_prefix}_quanv_nn_config.json"

    def get_encoder_filename(self, filename_prefix: str) -> str:
        """Get encoder filename.

        :param str filename_prefix: prefix of filename
        :return str: encoder filename
        """
        return f"{filename_prefix}_encoder.pickle"

    def get_decoder_filename(self, filename_prefix: str) -> str:
        """Get decoder filename.

        :param str filename_prefix: prefix of filename
        :return str: decoder filename
        """
        return f"{filename_prefix}_decoder.pickle"

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
        classical_cnn_filename = self.get_classical_cnn_filename(
            filename_prefix=filename_prefix
        )
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
        config_filename = self.get_qnn_config_filename(filename_prefix=filename_prefix)
        quanv_output_dir = os.path.join(output_dir, "quanv")
        if not os.path.exists(quanv_output_dir):
            os.makedirs(quanv_output_dir)
        config_path = os.path.join(quanv_output_dir, config_filename)
        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=4)

        # Save the encoder.
        encoder_filename = self.get_encoder_filename(filename_prefix=filename_prefix)
        encoder_path = os.path.join(quanv_output_dir, encoder_filename)
        with open(encoder_path, "wb") as encoder_file:
            pickle.dump(
                self.quanv_encoder,
                encoder_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # Save the decoder.
        decoder_filename = self.get_decoder_filename(filename_prefix=filename_prefix)
        decoder_path = os.path.join(quanv_output_dir, decoder_filename)
        with open(decoder_path, "wb") as decoder_file:
            pickle.dump(
                self.quanv_decoder,
                decoder_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # Save each QuanvFilter.
        for index, quanv_filter in enumerate(self.quanv_layer.quanv_filters):
            quanv_filter_filename_prefix = f"{filename_prefix}_{index}"
            quanv_filter.save(
                output_dir=quanv_output_dir,
                filename_prefix=quanv_filter_filename_prefix,
            )

    def load(self, input_dir: str, filename_prefix):
        """Load the QNN.

        :param str input_dir: path to input dir
        :param str filename_prefix: prefix of input files
        """
        # Load the ClassicalCNN.
        classical_cnn_input_dir = os.path.join(input_dir, "classical_cnn")
        classical_cnn_filename = self.get_classical_cnn_filename(
            filename_prefix=filename_prefix
        )
        classical_cnn_path = os.path.join(
            classical_cnn_input_dir, classical_cnn_filename
        )
        self.classical_cnn.load_state_dict(
            torch.load(classical_cnn_path, weights_only=True)
        )

        # Load the QNN config.
        config_filename = self.get_qnn_config_filename(filename_prefix=filename_prefix)
        quanv_input_dir = os.path.join(input_dir, "quanv")
        config_path = os.path.join(quanv_input_dir, config_filename)
        with open(config_path) as config_file:
            config = json.load(config_file)
        self.in_dim = config["in_dim"]
        self.num_classes = config["num_classes"]
        self.quanv_kernel_size = tuple(config["quanv_kernel_size"])
        self.quanv_num_filters = config["quanv_num_filters"]
        self.quanv_padding_mode = config["quanv_padding_mode"]
        # Reset the QuanvLayer config.
        self.quanv_layer.kernel_size = self.quanv_kernel_size
        self.quanv_layer.num_filters = self.quanv_num_filters
        self.quanv_layer.padding_mode = self.quanv_padding_mode

        # Load the encoder.
        encoder_filename = self.get_encoder_filename(filename_prefix=filename_prefix)
        encoder_path = os.path.join(quanv_input_dir, encoder_filename)
        with open(encoder_path, "rb") as encoder_file:
            self.encoder = pickle.load(encoder_file)

        # Load the decoder.
        decoder_filename = self.get_decoder_filename(filename_prefix=filename_prefix)
        decoder_path = os.path.join(quanv_input_dir, decoder_filename)
        with open(decoder_path, "rb") as decoder_file:
            self.decoder = pickle.load(decoder_file)

        # Load each QuanvFilter.
        for index, quanv_filter in enumerate(self.quanv_layer.quanv_filters):
            quanv_filter_filename_prefix = f"{filename_prefix}_{index}"
            quanv_filter.load(
                input_dir=quanv_input_dir,
                filename_prefix=quanv_filter_filename_prefix,
            )
