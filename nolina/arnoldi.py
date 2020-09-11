"""Arnoldi method."""

import numpy as np
import numpy.linalg as la
from nolina import random


def arnoldi(A, b, j, x=None, random_state=None):
    """Arnoldi method (modified Gram-Schmidt). Algorithm 23."""
    x = random.get_start_vector(d=A.shape[0], y0=x, random_state=random_state)
    r = b - A @ x
    v = np.empty((j, A.shape[0]))
    v[0] = r / la.norm(r)
    for k in range(1, j):
        w = A @ v[k - 1]
        for l in range(k):
            w -= np.dot(w, v[l]) * v[l]
        h = la.norm(w)
        if 0 == h:
            break
        else:
            v[k] = w / h
    return v
