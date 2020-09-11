"""Test power method."""

import numpy as np
import numpy.linalg as la
from nolina import power, random


def test_power_method(rng, seed):
    print(rng.uniform())

    A = random.random_spsd_matrix(d=10, random_state=rng)
    w_power, v_power, sigma_power = power.power_method(A, niter=100, random_state=rng)

    w_eigh, v_eigh = la.eigh(A)

    np.testing.assert_allclose(v_eigh[:, -1], v_power)
    np.testing.assert_allclose(w_eigh[-1], w_power)
