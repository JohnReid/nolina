"""Test power method."""

import numpy as np
import numpy.linalg as la
from nolina import power


def test_power_method(rng, seed):
    print(rng.uniform())

    A = power.random_matrix(d=10, random_state=rng)
    A = A @ A.T  # Make symmetric
    v_power = power.power_method(A, niter=100, random_state=rng)

    w, v_eigh = la.eigh(A)

    np.testing.assert_allclose(v_eigh[:, -1], v_power)
