import pytest
from nolina import random


def test_zero_division(rng, seed):
    with pytest.raises(ValueError):
        d = 10
        v = random.random_vector(d, random_state=rng)
        v = random.get_start_vector(d, y0=v, random_state=rng)
