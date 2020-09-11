from codetiming import Timer
import logging
import pytest
import numpy as np
from nolina import random, conjugate_gradient as cg

_logger = logging.getLogger(__name__)
ds = [9, 21, 55]


@pytest.mark.parametrize("d", ds)
def test_steepest_descent(d, rng, seed):
    minimiser = cg.SteepestDescent(A=random.random_spsd_matrix(d=d, random_state=rng),
                                   b=random.random_vector(d=d, random_state=rng))
    with Timer(text='Steepest descent minimiser done in {:.4f} seconds', logger=_logger.info):
        x_star = minimiser()
    _logger.info('Steepest descent minimiser took %d iterations.', minimiser.niter)
    np.testing.assert_allclose(minimiser.A @ x_star, minimiser.b)


@pytest.mark.parametrize("d", ds)
def test_conjugate_gradient_preliminary(d, rng, seed):
    minimiser = cg.ConjugateGradientPreliminary(A=random.random_spsd_matrix(d=d, random_state=rng),
                                                b=random.random_vector(d=d, random_state=rng))
    with Timer(text='Conjugate gradient preliminary minimiser done in {:.4f} seconds', logger=_logger.info):
        x_star = minimiser()
    _logger.info('Conjugate gradient preliminary minimiser took %d iterations.', minimiser.niter)
    np.testing.assert_allclose(minimiser.A @ x_star, minimiser.b)


@pytest.mark.parametrize("d", ds)
def test_conjugate_gradient(d, rng, seed):
    minimiser = cg.ConjugateGradient(A=random.random_spsd_matrix(d=d, random_state=rng),
                                     b=random.random_vector(d=d, random_state=rng))
    with Timer(text='Conjugate gradient minimiser done in {:.4f} seconds', logger=_logger.info):
        x_star = minimiser()
    _logger.info('Conjugate gradient minimiser took %d iterations.', minimiser.niter)
    np.testing.assert_allclose(minimiser.A @ x_star, minimiser.b)
