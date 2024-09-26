from .base_decoder import BaseDecoder


class OneSumDecoder(BaseDecoder):
    """One sum decoder clsss"""

    def __init__(self):
        """Initialise this decoder."""
        pass

    def decode(self, counts: dict) -> int:
        """Decode the counts from the result of an execution of qiskit.QuantumCircuit by summing ones.

        :param dict counts: result of execution of qiskit.QuantumCircuit
        :return int: decoded scalar
        """
        # Sort the resuly by the frequency.
        sorted_counts = dict(sorted(counts.items(), key=lambda item: -item[1]))
        # Get the most likely result.
        most_likely_result = list(sorted_counts.keys())[0]
        # Count the number of ones.
        num_ones = most_likely_result.count("1")
        return num_ones
