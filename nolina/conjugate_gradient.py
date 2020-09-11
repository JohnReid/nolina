"""Conjugate gradient methods."""

import numpy as np
import numpy.linalg as la
from dataclasses import dataclass
from typing import List


@dataclass
class GenericMinimiser:
    """Algorithm 19."""

    A: List[float]
    b: List[float]

    def __call__(self, x=None, eps=1e-12):
        self.niter = 0
        if x is None:
            x = np.zeros_like(self.b)
        r = self.b - self.A @ x
        p = self._initial_direction(x, r)
        while la.norm(r) > eps:
            x, p, r = self._step(x, p, r)
        return x

    def _step(self, x, p, r):
        alpha = np.dot(r, p) / np.dot(self.A @ p, p)
        x = x + alpha * p
        r = self.b - self.A @ x
        p = self._choose_direction(x, p, r)
        self.niter += 1
        return x, p, r

    def _initial_direction(self, x, r):
        return self._choose_direction(x, np.zeros_like(x), r)

    def _choose_direction(self, x, p, r):
        raise NotImplementedError()


class SteepestDescent(GenericMinimiser):
    """Algorithm 20."""

    def _choose_direction(self, x, p, r):
        return r


class ConjugateGradientPreliminary(GenericMinimiser):
    """Algorithm 21."""

    def _initial_direction(self, x, r):
        return r

    def _choose_direction(self, x, p, r):
        beta = - np.dot(self.A @ p, r) / np.dot(self.A @ p, p)
        return r + beta * p


class ConjugateGradient(GenericMinimiser):
    """Algorithm 22."""

    def _step(self, x, p, r):
        t = self.A @ p
        old_r2 = np.dot(r, r)
        alpha = old_r2 / np.dot(t, p)
        x = x + alpha * p
        r = r - alpha * t
        beta = np.dot(r, r) / old_r2
        p = r + beta * p
        self.niter += 1
        return x, p, r

    def _initial_direction(self, x, r):
        return r
