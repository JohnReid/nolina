from codetiming import Timer
import logging
import pytest
import numpy as np
from nolina import random, arnoldi

_logger = logging.getLogger(__name__)
js = [9, 10, 11]


def close_to_orthonormal(v):
    return (np.eye(v.shape[0]) == v @ v.T).allclose()


@pytest.mark.parametrize("j", js)
def test_arnoldi(j, rng, seed):
    d = 11
    A = random.random_spsd_matrix(d=d, random_state=rng)
    b = random.random_vector(d=d, random_state=rng)

    with Timer(text='Arnoldi done in {:.4f} seconds', logger=_logger.info):
        v = arnoldi.arnoldi(A, b, j=j)

    np.testing.assert_almost_equal(np.eye(v.shape[0]), v @ v.T)
