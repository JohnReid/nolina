"""Randomised linear algebra."""

import numpy.linalg as la


def normalise(v):
    norm = la.norm(v)
    return v if 0 == norm else v / norm
