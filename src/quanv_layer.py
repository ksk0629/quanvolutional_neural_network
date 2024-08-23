import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from quanv_filter import QuanvFilter


class QuanvLayer:
    """Quanvolutional layer class."""

    def __init__(
        self,
        kernel_size: tuple[int, int],
        num_filters: int,
        padding_mode: str | None = "constant",
    ):
        """Initialise the instance.

        :param tuple[int, int] kernel_size: kernel size
        :param int num_filters: number of filters
        :param str | None padding_mode: padding mode (see the document of torch.nn.functional.pad), defaults to "constant"
        """
        # Store the arguments to class variables.
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.padding_mode = padding_mode

        # Define constants.
        self.__batch_data_dim = 4

        # Get the quanvolutional filters.
        self.quanv_filters = [
            QuanvFilter(self.kernel_size) for _ in range(self.num_filters)
        ]

    def run_for_batch(self, batch_data: torch.Tensor, shots: int) -> torch.Tensor:
        """Run the circuit with the given dataset.

        :param torch.Tensor batch_data: batch_data whose shape must be [batch size, channel, height, width]
        :param int shots: number of shots
        :return torch.Tensor: processed batch_data whose shape must be [batch size, channel, height, width]
        """
        # Check the dataset shape.
        if batch_data.ndim != self.__batch_data_dim:
            msg = f"""
                The dimension of the batch_data must be {self.__batch_data_dim},
                which is [batch size, channel, height, width].
            """
            raise ValueError(msg)

        # Process all data.
        all_outputs = torch.stack(
            [
                self.run_single_channel(data=data, shots=shots)
                for data in tqdm(batch_data, leave=True, desc="Dataset")
            ]
        )

        return all_outputs

    def run_single_channel(self, data: torch.Tensor, shots: int) -> torch.Tensor:
        """Run the circuit with a single channel image.

        :param torch.Tensor data: single channel image data
        :param int shots: number of shots
        :return torch.Tensor: processed single channel image data
        """
        # Get only one data from the data.
        # This is a valid operation as this method assumes that the data is a single channel data.
        data = data[0]

        # Perform padding accodring to the mode to make the output shape as same as the input.
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
        padded_data = F.pad(
            data,
            pad,
            mode=self.padding_mode,
        )

        # Make the sliding window data.
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

        # Reshape the sliding window data to feed to the quanvolutional filters.
        reshaped_sliding_window_data = torch.reshape(
            sliding_window_data, (-1, self.kernel_size[0] * self.kernel_size[1])
        )

        # >>> Numpy computing zone >>>
        # Conver the sliding window data from torch.Tensor to numpy.
        reshaped_strided_data_np = reshaped_sliding_window_data.detach().cpu().numpy()
        # Make the initial outputs data as numpy.ndarray.
        outputs = np.empty([len(self.quanv_filters), *data.shape])
        for index, quanvolutional_filter in enumerate(
            tqdm(self.quanv_filters, leave=False, desc="Filters")
        ):
            # Vectorise quanvolutional_filter.run function to make it quick.
            vectorized_quanvolutional_filter_run = np.vectorize(
                quanvolutional_filter.run, signature="(n),()->()"
            )
            # Perform each quanvolutional filter.
            outputs[index, :, :] = vectorized_quanvolutional_filter_run(
                reshaped_strided_data_np, shots
            ).reshape(data.shape)
        outputs = torch.Tensor(outputs)
        # <<< Numpy computing zone <<<

        return outputs

    def run_for_batch_and_save(
        self, batch_data: torch.Tensor, shots: int, filename: str
    ):
        """Run the circuit with the given batch data and save the result.

        :param torch.Tensor batch_data: batch_data
        :param int shots: number of shots
        :param str filename: output path
        """
        outputs = self.run_for_batch(batch_data=batch_data, shots=shots)
        torch.save(outputs, filename)
