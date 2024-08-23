import numpy as np


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
