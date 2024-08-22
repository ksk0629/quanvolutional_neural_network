import numpy as np
import qiskit
import qiskit_aer
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from quanvolutional_filter import QuanvolutionalFilter


class QuanvolutionalLayer:
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
        self.__dataset_dimension = 4

        # Get the quanvolutional filters.
        self.quanvolutional_filters = [
            QuanvolutionalFilter(self.kernel_size) for _ in range(self.num_filters)
        ]

        self.backend = qiskit_aer.AerSimulator()
        self.backend.set_options(
            max_parallel_threads=0, max_parallel_experiments=0, max_parallel_shots=0
        )

    def run_for_batch(self, batch_data: torch.Tensor, shots: int) -> torch.Tensor:
        """Run the circuit with the given dataset.

        :param torch.Tensor batch_data: batch_data whose shape must be [batch size, channel, height, width]
        :param int shots: number of shots
        :return torch.Tensor: processed batch_data whose shape must be [batch size, channel, height, width]
        """
        # Check the dataset shape.
        if batch_data.ndim != self.__dataset_dimension:
            msg = f"""
                The dimension of the dataset must be {self.__dataset_dimension},
                which is [batch size, channel, height, width].
            """
            raise ValueError(msg)

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

        # Perform padding to make the output shape as same as the input  accodring to the mode.
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

        # Make the strided data.
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
        strided_data = torch.as_strided(padded_data, size=new_shape, stride=new_stride)

        # Reshape the strided data to feed to the quanvolutional filters.
        reshaped_strided_data = torch.reshape(
            strided_data, (-1, self.kernel_size[0] * self.kernel_size[1])
        )

        # Prepare the output data.
        outputs = np.empty([len(self.quanvolutional_filters), *data.shape])

        # Convert the strided data into list.
        reshaped_strided_data_list = reshaped_strided_data.detach().cpu().tolist()

        # Perform the filters to each window data.
        for index, window_data in enumerate(
            tqdm(reshaped_strided_data_list, leave=False, desc="window")
        ):

            # Load encoded data to each circuit.
            for quanvolutional_filter in self.quanvolutional_filters:
                encoded_data = QuanvolutionalLayer.encode_with_threshold(
                    np.array(window_data)
                )
                quanvolutional_filter.load_data(encoded_data)

            # Run the circuits parallely.
            circuits = [
                quanvolutional_filter.circuit
                for quanvolutional_filter in self.quanvolutional_filters
            ]
            transpiled_circuits = qiskit.transpile(circuits, backend=self.backend)
            results = self.backend.run(transpiled_circuits).result()
            counts = results.get_counts()

            # Decode the results.
            decoded_datum = [
                QuanvolutionalLayer.decode_by_summing_ones(count) for count in counts
            ]

            # Store the decoded data to the output data.
            row_index = index // data.shape[1]
            column_index = index % data.shape[1]
            for filter_index, decoded_data in enumerate(decoded_datum):
                outputs[filter_index, row_index, column_index] = decoded_data

        outputs = torch.Tensor(outputs)
        return outputs

    def run_for_batch_and_save(
        self, batch_data: torch.Tensor, shots: int, filename: str
    ):
        """Run the circuit with the given batch data and save the result.

        :param torch.Tensor batch_data: batch_data
        :param int shots: number of shots
        :param str filename: output path
        """
        outputs = self.run_for_batch(dataset=batch_data, shots=shots)
        torch.save(outputs, filename)

    @staticmethod
    def encode_with_threshold(data: np.ndarray, threshold: float = 1) -> np.ndarray:
        """Encode the given data according to the threshold. This method is suggested in the original paper.

        :param np.ndarray data: original data
        :param float threshold: threshold to encode
        :return np.ndarray: encoded data
        """
        flatten_data = data.flatten()
        encode_flags = np.where(flatten_data >= threshold, 1, 0).astype(np.float64)
        quantum_state = 1
        for encode_flag in encode_flags:
            encoded_state = np.array([1, 0]) if encode_flag == 0 else np.array([0, 1])
            quantum_state = np.kron(quantum_state, encoded_state)

        return quantum_state

    @staticmethod
    def decode_by_summing_ones(counts: dict) -> int:
        """Decode the measured result to the number of ones in the result.

        :param dict counts: result of running the circuit
        :return int: the number of ones in the most likely result
        """
        # Sort the resuly by the frequency.
        sorted_counts = dict(sorted(counts.items(), key=lambda item: -item[1]))
        # Get the most likely result.
        most_likely_result = list(sorted_counts.keys())[0]
        # Count the number of ones.
        num_ones = most_likely_result.count("1")

        return num_ones
