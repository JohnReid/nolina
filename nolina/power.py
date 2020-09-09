"""Implement the power method."""

import numpy as np
from scipy import stats


_standard_normal = stats.norm()


def random_matrix(d: int, random_state=None):
    return _standard_normal.rvs(size=(d, d), random_state=random_state)


def normalise(v):
    norm = np.linalg.norm(v)
    return v if 0 == norm else v / norm


def power_method(A, niter, random_state=None):
    d = A.shape[0]
    y = np.empty((niter + 1, d))
    y[0] = _standard_normal.rvs(size=d, random_state=random_state)
    for m in range(1, niter + 1):
        y[m] = normalise(A @ y[m - 1])
    return y[niter]
