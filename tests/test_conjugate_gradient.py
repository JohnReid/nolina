import numpy as np
from nolina import random, conjugate_gradient as cg


def test_steepest_descent(rng, seed):
    d = 11
    minimiser = cg.SteepestDescent(A=random.random_spsd_matrix(d=d, random_state=rng),
                                   b=random.random_vector(d=d, random_state=rng))
    x_star = minimiser()
    np.testing.assert_allclose(minimiser.A @ x_star, minimiser.b)
