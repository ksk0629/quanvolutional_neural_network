import random

import numpy as np
import pytest
import torch

from src.utils import fix_seed


class TestFixSeed:
    """Test class for src.utils.fix_seed function."""

    @pytest.mark.utils
    @pytest.mark.parametrize("seed", [129, 42, 57])
    def test_same_fix_seeds_random(self, seed: int):
        """Normal test for the random module.

        :param int seed: random seed
        """
        fix_seed(seed)
        value_1 = random.random()
        fix_seed(seed)
        value_2 = random.random()
        assert value_1 == value_2

    @pytest.mark.utils
    @pytest.mark.parametrize("seed", [129, 42, 57])
    def test_same_fix_seeds_numpy(self, seed: int):
        """Normal test for numpy module.

        :param int seed: random seed
        """
        fix_seed(seed)
        value_1 = np.random.rand()
        fix_seed(seed)
        value_2 = np.random.rand()
        assert value_1 == value_2

    @pytest.mark.utils
    @pytest.mark.parametrize("seed", [129, 42, 57])
    def test_same_fix_seeds_torch(self, seed: int):
        """Normal test for torch module.

        :param int seed: random seed
        """
        fix_seed(seed)
        value_1 = torch.rand(1)
        fix_seed(seed)
        value_2 = torch.rand(1)
        assert value_1 == value_2
