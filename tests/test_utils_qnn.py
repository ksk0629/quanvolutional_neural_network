import numpy as np
import pytest

from src.utils_qnn import encode_with_threshold


class TestEncodeWithThreshold:
    """Test class for src.utils_qnn.encode_with_threshold function."""

    @classmethod
    def setup_class(self):
        """Setup this test class."""
        self.num_qubits = 5
        self.normal_data = np.arange(self.num_qubits)
        self.normal_threshold = 1

    @pytest.mark.utils
    def test_use_normal_args(self):
        """Normal test with normal arguments;
        The return value
        - is np.ndarray.
        - has one-dimension.
        - has 2**self.num_qubits elements.
        """
        quantum_states = encode_with_threshold(self.normal_data, self.normal_threshold)
        assert isinstance(quantum_states, np.ndarray)
        assert len(quantum_states.shape) == 1
        assert quantum_states.shape[0] == 2**self.num_qubits

    @pytest.mark.utils
    @pytest.mark.parametrize("abnormal_data", [1, 1.1, [0, 1, 2, 3, 4], "12345", True])
    def test_with_abnormal_data(self, abnormal_data):
        """Abnormal test;
        AttributeError occurs.
        """
        with pytest.raises(AttributeError):
            encode_with_threshold(abnormal_data, self.normal_threshold)

    @pytest.mark.utils
    @pytest.mark.parametrize("abnormal_threshold", ["12345"])
    def test_with_abnormal_threshold(self, abnormal_threshold):
        """Abnormal test;
        TypeError occurs.
        """
        with pytest.raises(TypeError):
            encode_with_threshold(self.normal_data, abnormal_threshold)
