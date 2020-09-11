"""Implement the power method."""

import numpy as np
import numpy.linalg as la
import nolina
from nolina import random


def power_method(A, niter, y0=None, random_state=None):
    """Power iteration method. Definition 5.12."""
    y0 = random.get_start_vector(d=A.shape[0], y0=y0, random_state=random_state)
    x0 = nolina.normalise(y0)
    for _ in range(niter):
        y0 = A @ x0
        sigma = -1 if np.dot(y0, x0) < 0 else 1
        y0 *= sigma
        x0 = nolina.normalise(y0)
    return la.norm(y0), x0, sigma
