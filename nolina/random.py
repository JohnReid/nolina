"""Random matrices."""

from scipy import stats

_standard_normal = stats.norm()


def get_start_vector(d, y0=None, random_state=None):
    if y0 is None:
        y0 = _standard_normal.rvs(size=d, random_state=random_state)
    else:
        if y0.shape != d:
            raise ValueError(f'Expecting y0 to have shape {d}')
    return y0


def random_vector(d: int, random_state=None):
    return _standard_normal.rvs(size=d, random_state=random_state)


def random_matrix(d: int, random_state=None):
    return _standard_normal.rvs(size=(d, d), random_state=random_state)


def random_spsd_matrix(d: int, random_state=None):
    A = random_matrix(d=d, random_state=random_state)
    return A @ A.T
