"""Conjugate gradient methods."""

import numpy as np
import numpy.linalg as la
from dataclasses import dataclass
from typing import List


@dataclass
class GenericMinimiser:

    A: List[float]
    b: List[float]

    def __call__(self, x=None, eps=1e-8):
        if x is None:
            x = np.zeros_like(self.b)
        r = self.b - self.A @ x
        while la.norm(r) > eps:
            x, r = self._step(x, r)
        return x

    def _step(self, x, r):
        p = self._choose_direction(x, r)
        alpha = np.dot(r, p) / np.dot(self.A @ p, p)
        x = x + alpha * p
        r = self.b - self.A @ x
        return x, r

    def _choose_direction(self, x, r):
        raise NotImplementedError()


class SteepestDescent(GenericMinimiser):
    def _choose_direction(self, x, r):
        return r
