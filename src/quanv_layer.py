import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from decoders.base_decoder import BaseDecoder
from encoders.base_encoder import BaseEncoder
from quanv_filter import QuanvFilter


class QuanvLayer:
    """Quanvolutional layer class."""

    def __init__(
        self,
        kernel_size: tuple[int, int],
        num_filters: int,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        padding_mode: str | None = "constant",
        is_lookup_mode: bool = True,
    ):
        """Initialise the instance.

        :param tuple[int, int] kernel_size: kernel size
        :param int num_filters: number of filters
        :param BaseEncoder encoder: encoder
        :param BaseDecoder decoder: decoder
        :param str | None padding_mode: padding mode (see the document of torch.nn.functional.pad), defaults to "constant"
        :param bool is_lookup_mode: if it is look-up mode, defaults to True
        """
        # Store the arguments to class variables.
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.encoder = encoder
        self.decoder = decoder
        self.padding_mode = padding_mode
        self.is_lookup_mode = is_lookup_mode

        # Define constant.
        self.__BATCH_DATA_DIM = 4

        # Create the quanvolutional filters.
        self.quanv_filters = [
            QuanvFilter(self.kernel_size) for _ in range(self.num_filters)
        ]

    def run(self, batch_data: torch.Tensor, shots: int) -> torch.Tensor:
        """Apply the quanvolutional filters to the given batch data.

        :param torch.Tensor batch_data: batch_data whose shape must be [batch size, channel, height, width]
        :param int shots: number of shots
        :return torch.Tensor: processed batch_data whose shape must be [batch size, channel, height, width]
        """
        # Check the dataset shape.
        if batch_data.ndim != self.__BATCH_DATA_DIM:
            msg = f"""
                The dimension of the batch_data must be {self.__BATCH_DATA_DIM},
                which is [batch size, channel, height, width].
            """
            raise ValueError(msg)

        # Set the appropriate function according to the mode.
        if self.is_lookup_mode:
            # Get all possible input patterns.
            possible_inputs = self.encoder.get_all_input_patterns(
                num_qubits=self.kernel_size[0] * self.kernel_size[1]
            )
            # Set each look-up table.
            [
                quanv_filter.set_lookup_table(
                    encoding_method=self.encoder.encode,
                    decoding_method=self.decoder.decode,
                    shots=shots,
                    input_patterns=possible_inputs,
                )
                for quanv_filter in self.quanv_filters
            ]
            _run = self.run_single_channel_with_lookup_tables
        else:
            _run = lambda data: self.run_single_channel(data=data, shots=shots)

        # Make the empty outputs.
        batch_size, _, height, width = batch_data.size()
        num_channels_output = len(batch_data[0]) * self.num_filters
        all_outputs = torch.empty((batch_size, num_channels_output, height, width))
        # Process all data.
        for data_index, data in enumerate(tqdm(batch_data, leave=True, desc="Dataset")):
            for channel_index, channel in enumerate(data):
                current_channel_index_start = channel_index * self.num_filters
                current_channel_index_end = (
                    channel_index * self.num_filters + self.num_filters
                )
                all_outputs[
                    data_index,
                    current_channel_index_start:current_channel_index_end,
                    :,
                    :,
                ] = _run(data=channel)

        return all_outputs

    def get_sliding_window_data(self, data: torch.Tensor) -> torch.Tensor:
        """Get the sliding window data.

        :param torch.Tensor data: input data
        :return torch.Tensor: data whose each entry is sliding window
        """
        # Perform padding according to the mode to make the output shape as same as the input.
        padded_data = self.pad(data)

        # Make the sliding window data for each entry that is a sliding window.
        new_shape = (
            padded_data.size(0) - self.kernel_size[0] + 1,
            padded_data.size(1) - self.kernel_size[1] + 1,
        ) + self.kernel_size
        new_stride = (
            padded_data.stride(0),
            padded_data.stride(1),
            padded_data.stride(0),
            padded_data.stride(1),
        )
        sliding_window_data = torch.as_strided(
            padded_data, size=new_shape, stride=new_stride
        )

        return sliding_window_data

    def pad(self, data: torch.Tensor) -> torch.Tensor:
        """Pad the input data.

        :param torch.Tensor data: input data
        :return torch.Tensor: padded data
        """
        # Get the padding sizes.
        padding_size_left = self.kernel_size[1] // 2
        padding_size_right = self.kernel_size[1] // 2
        padding_size_top = self.kernel_size[0] // 2
        padding_size_bottom = self.kernel_size[0] // 2
        pad = (
            padding_size_left,
            padding_size_right,
            padding_size_top,
            padding_size_bottom,
        )
        # Pad the data.
        padded_data = F.pad(
            data,
            pad,
            mode=self.padding_mode,
        )
        return padded_data

    def run_single_channel(self, data: torch.Tensor, shots: int) -> torch.Tensor:
        """Run the circuit with a single channel image data.

        :param torch.Tensor data: single channel image data
        :param int shots: number of shots
        :return torch.Tensor: processed single channel image data
        """
        # Get the sliding window data.
        sliding_window_data = self.get_sliding_window_data(data)

        # Reshape the sliding window data fed to the quanvolutional filters.
        reshaped_sliding_window_data = torch.reshape(
            sliding_window_data, (-1, self.kernel_size[0] * self.kernel_size[1])
        )

        # >>> Numpy computing zone >>>
        # Convert the sliding window data from torch.Tensor to numpy.
        reshaped_strided_data_np = reshaped_sliding_window_data.detach().cpu().numpy()
        # Make the initial output data as numpy.ndarray.
        outputs = np.empty([len(self.quanv_filters), data.shape[-2], data.shape[-1]])

        for index, quanvolutional_filter in enumerate(
            tqdm(self.quanv_filters, leave=False, desc="Filters")
        ):
            # Vectorise quanvolutional_filter.run function to make it fast.
            vectorized_quanvolutional_filter_run = np.vectorize(
                quanvolutional_filter.run, signature="(n),(),(),()->()"
            )
            # Apply each quanvolutional filter.
            outputs[index, :, :] = vectorized_quanvolutional_filter_run(
                reshaped_strided_data_np,
                self.encoder.encode,
                self.decoder.decode,
                shots,
            ).reshape(data.shape)
        outputs = torch.Tensor(outputs)
        # <<< Numpy computing zone <<<

        return outputs

    def run_single_channel_with_lookup_tables(self, data: torch.Tensor) -> torch.Tensor:
        """Use the look-up tables to process a single channel image.

        :param torch.Tensor data: single channel image data
        :return torch.Tensor: processed single channel image data
        """
        # Get the sliding window data.
        sliding_window_data = self.get_sliding_window_data(data)

        # Reshape and convert the sliding window data into list.
        reshaped_sliding_window_data = torch.reshape(
            sliding_window_data, (-1, self.kernel_size[0] * self.kernel_size[1])
        ).tolist()

        # Encode the window data to one of the keys of the look-up tables.
        encoded_slising_window_data = [
            self.encoder.convert_to_input_boundary(small_window_data)
            for small_window_data in reshaped_sliding_window_data
        ]

        # Make the output data using the look-up tables.
        outputs = np.empty([len(self.quanv_filters), data.shape[-2], data.shape[-1]])
        for index, quanvolutional_filter in enumerate(
            tqdm(self.quanv_filters, leave=False, desc="Filters (Look-up tables)")
        ):
            output = np.array(
                [
                    quanvolutional_filter.lookup_table[small_window]
                    for small_window in encoded_slising_window_data
                ]
            )
            outputs[index, :, :] = output.reshape(data.shape)
        outputs = torch.Tensor(outputs)

        return outputs
