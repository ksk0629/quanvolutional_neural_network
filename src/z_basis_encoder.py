import itertools

import numpy as np

from base_encoder import BaseEncoder


class ZBasisEncoder(BaseEncoder):
    """Z-basis states encoder class"""

    def __init__(self, threshold: float = 0):
        """Initialise the encoder.

        :param float threshold: threshold to encode data to the |0> or |1>.
        """
        self.threshold = threshold
        self.KET_0 = np.array([1, 0]).astype(np.float64)
        self.KET_1 = np.array([0, 1]).astype(np.float64)

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode the data to the product state of z-basis states.

        :param np.ndarray data: input data
        :return np.ndarray: encoded data, which is quantum state
        """
        # Flatten the data.
        flatten_data = data.flatten().tolist()

        # Generate the encoded data, which is a quantum state.
        quantum_state = 1
        for one_dim_data in flatten_data:
            encoded_state = self.encode_one_data(one_dim_data)
            quantum_state = np.kron(quantum_state, encoded_state)

        return quantum_state

    def encode_one_data(self, one_dimensional_data: float) -> np.ndarray:
        """Encode one-dimensional data to one of the z-basis states.

        :param float one_dimensional_data: one dimensional data
        :return np.ndarray: one of z-basis states
        """
        if one_dimensional_data > self.threshold:
            encoded_data = self.KET_1
        else:
            encoded_data = self.KET_0

        return encoded_data

    def get_all_input_patterns(
        self, num_qubits: int
    ) -> list[tuple[int, int] | tuple[float, float]]:
        """Get all input patterns.
        Remark there are finite patterns corresponding all encoded patterns
        because this encoder encodes data using threshold.

        :param int num_qubits: number of qubits
        :return list[tuple[int, int] | tuple[float, float]]: comprehensive input patterns
        """
        return list(
            itertools.product(
                [self.threshold + 1, self.threshold],
                repeat=num_qubits,
            )
        )
