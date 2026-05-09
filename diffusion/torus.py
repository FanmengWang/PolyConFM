import numpy as np
import tqdm
import os


def p(x, sigma, N=10):
    """Compute wrapped Gaussian density by truncated periodic summation.

    Args:
        x: Angle values (radians) where the density is evaluated.
        sigma: Diffusion noise scale(s) for wrapped Gaussian kernels.
        N: Number of wrap terms on each side of zero.

    Returns:
        Approximated wrapped density values with broadcasted shape.
    """
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


def grad(x, sigma, N=10):
    """Compute derivative numerator used for wrapped score values.

    Args:
        x: Angle values (radians).
        sigma: Diffusion noise scale(s).
        N: Number of wrap terms on each side of zero.

    Returns:
        Truncated-sum derivative term aligned with p(x, sigma, N).
    """
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += (x + 2 * np.pi * i) / sigma ** 2 * np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi

if os.path.exists('.p.npy'):
    p_ = np.load('.p.npy')
    score_ = np.load('.score.npy')
else:
    p_ = p(x, sigma[:, None], N=100)
    np.save('.p.npy', p_)

    score_ = grad(x, sigma[:, None], N=100) / p_
    np.save('.score.npy', score_)


def score(x, sigma):
    """Lookup wrapped-torsion score from precomputed grids.

    Args:
        x: Input torsion angles (radians).
        sigma: Noise scale values (radians) matching x by broadcasting.

    Returns:
        Score values d/dx log p(x | sigma) from cached lookup tables.
    """
    x = (x + np.pi) % (2 * np.pi) - np.pi
    sign = np.sign(x)
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return -sign * score_[sigma, x]


def p(x, sigma):
    """Lookup wrapped-torsion density from precomputed grids.

    Args:
        x: Input torsion angles (radians).
        sigma: Noise scale values (radians) matching x by broadcasting.

    Returns:
        Cached density values corresponding to (x, sigma) pairs.
    """
    x = (x + np.pi) % (2 * np.pi) - np.pi
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return p_[sigma, x]


def sample(sigma):
    """Sample wrapped Gaussian torsion perturbations.

    Args:
        sigma: Noise scale tensor/array whose shape defines sample shape.

    Returns:
        Wrapped angle samples in [-pi, pi] with the same shape as sigma.
    """
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out


score_norm_ = score(
    sample(sigma[None].repeat(10000, 0).flatten()),
    sigma[None].repeat(10000, 0).flatten()
).reshape(10000, -1)
score_norm_ = (score_norm_ ** 2).mean(0)


def score_norm(sigma):
    """Return precomputed score normalization for given sigma values.

    Args:
        sigma: Noise scale tensor/array in radians.

    Returns:
        Expected squared-score statistics indexed from cached table.
    """
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return score_norm_[sigma]
