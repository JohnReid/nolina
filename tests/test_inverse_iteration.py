"""Test inverse iteration method."""

import numpy as np
import numpy.linalg as la
from nolina import inverse_iteration, random


def test_inverse_iteration(rng, seed):
    A = random.random_spsd_matrix(d=10, random_state=rng)
    w_eigh, v_eigh = la.eigh(A)

    weighting = .99
    lambda_ = weighting * w_eigh[-1] + (1 - weighting) * w_eigh[-2]
    w_power = inverse_iteration.inverse_iteration(A, lambda_=lambda_, niter=100, random_state=rng)

    np.testing.assert_allclose(w_eigh[-1], w_power, rtol=0.01)
