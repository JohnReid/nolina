"""Implement the inverse iteration method from von Wielandt and Rayleigh."""

import numpy as np
import numpy.linalg as la
from scipy.linalg import lu_factor, lu_solve
from nolina import random, normalise


def inverse_iteration(A, lambda_, niter, y0=None, random_state=None):
    lu, piv = lu_factor(A - lambda_ * np.eye(A.shape[0]))
    return inverse_iteration_lu(lu, piv, lambda_, niter, y0=None, random_state=None)


def inverse_iteration_lu(lu, piv, lambda_, niter, y0=None, random_state=None):
    y0 = random.get_start_vector(d=lu.shape[0], y0=y0, random_state=random_state)
    x0 = normalise(y0)
    for _ in range(niter):
        y0 = lu_solve((lu, piv), x0)
        sigma = -1 if np.dot(y0, x0) < 0 else 1
        y0 *= sigma
        x0 = normalise(y0)
    return lambda_ + 1 / la.norm(y0)


def rayleigh_iteration(A, lambda_, niter, y0=None, random_state=None):
    y0 = random.get_start_vector(d=A.shape[0], y0=y0, random_state=random_state)
    x0 = normalise(y0)
    for _ in range(niter):
        lu, piv = lu_factor(A - lambda_ * np.eye(A.shape[0]))
        y0 = lu_solve((lu, piv), x0)
        sigma = -1 if np.dot(y0, x0) < 0 else 1
        y0 *= sigma
        x0 = normalise(y0)
        lambda_ = x0.T @ A @ x0
    return lambda_ + 1 / la.norm(y0)
