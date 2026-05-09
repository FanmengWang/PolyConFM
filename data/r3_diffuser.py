"""R^3 diffusion methods."""
import numpy as np
from scipy.special import gamma
import torch


class R3Diffuser:
    """VP-SDE diffuser class for translations."""

    def __init__(self, r3_conf):
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self._r3_conf = r3_conf
        self.min_b = r3_conf.min_b
        self.max_b = r3_conf.max_b

    def _scale(self, x):
        """Scale Cartesian coordinates into diffusion space.

        Args:
            x: Input coordinates in the original spatial scale.

        Returns:
            Coordinates scaled by the configured coordinate factor.
        """
        return x * self._r3_conf.coordinate_scaling

    def _unscale(self, x):
        """Map diffusion-space coordinates back to original scale.

        Args:
            x: Coordinates expressed in diffusion-scaled units.

        Returns:
            Coordinates restored to the original spatial scale.
        """
        return x / self._r3_conf.coordinate_scaling

    def b_t(self, t):
        """Evaluate the linear variance schedule at time t.

        Args:
            t: Scalar or array-like diffusion time in [0, 1].

        Returns:
            Schedule value beta(t) with the same broadcast shape as t.
        """
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return self.min_b + t*(self.max_b - self.min_b)

    def diffusion_coef(self, t):
        """Compute time-dependent diffusion coefficient.

        Args:
            t: Diffusion time value(s) in [0, 1].

        Returns:
            Diffusion coefficient value(s) sqrt(beta(t)).
        """
        return np.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        """Compute VP-SDE drift coefficient at time t.

        Args:
            x: Translation tensor/array at time t.
            t: Diffusion time value(s) in [0, 1].

        Returns:
            Drift term with the same shape as x.
        """
        return -1/2 * self.b_t(t) * x

    def sample_ref(self, n_samples: float=1):
        """Draw translation samples from the standard reference prior.

        Args:
            n_samples: Number of 3D translation vectors to generate.

        Returns:
            Array with shape [n_samples, 3] sampled from N(0, I).
        """
        return np.random.normal(size=(n_samples, 3))

    def marginal_b_t(self, t):
        """Compute the time-integrated beta schedule.

        Args:
            t: Scalar or array-like diffusion time in [0, 1].

        Returns:
            Integrated schedule value \int_0^t beta(s) ds.
        """
        return t*self.min_b + (1/2)*(t**2)*(self.max_b-self.min_b)

    def calc_trans_0(self, score_t, x_t, t, use_torch=True):
        """Estimate clean translations from noisy coordinates and score.

        Args:
            score_t: Score tensor/array evaluated at time t.
            x_t: Noisy translations at time t.
            t: Scalar or broadcastable diffusion time.
            use_torch: Whether to use torch.exp (True) or numpy.exp (False).

        Returns:
            Estimated clean translations x_0 in scaled coordinate space.
        """
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp if use_torch else np.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1/2*beta_t)

    def forward(self, x_t_1: np.ndarray, t: float, num_t: int):
        """Samples marginal p(x(t) | x(t-1)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_t_1 = self._scale(x_t_1)
        b_t = torch.tensor(self.marginal_b_t(t) / num_t).to(x_t_1.device)
        z_t_1 = torch.tensor(np.random.normal(size=x_t_1.shape)).to(x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def distribution(self, x_t, score_t, t, mask, dt):
        """Compute Gaussian reverse-step parameters for translations.

        Args:
            x_t: Current translations at time t.
            score_t: Predicted score at time t.
            t: Diffusion time in [0, 1].
            mask: Optional residue mask for selective updates.
            dt: Reverse-time step size.

        Returns:
            Tuple (mu, std) for the reverse transition distribution.
        """
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * np.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std

    def forward_marginal(self, x_0: np.ndarray, t: float):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_0 = self._scale(x_0)
        x_t = np.random.normal(
            loc=np.exp(-1/2*self.marginal_b_t(t)) * x_0,
            scale=np.sqrt(1 - np.exp(-self.marginal_b_t(t)))
        )
        score_t = self.score(x_t, x_0, t)
        x_t = self._unscale(x_t)
        return x_t, score_t

    def score_scaling(self, t: float):
        """Compute inverse-std scaling factor for translation score targets.

        Args:
            t: Diffusion time in [0, 1].

        Returns:
            Scalar or array of score normalization factors.
        """
        return 1 / np.sqrt(self.conditional_var(t))

    def reverse(
            self,
            *,
            x_t: np.ndarray,
            score_t: np.ndarray,
            t: float,
            dt: float,
            mask: np.ndarray=None,
            center: bool=True,
            noise_scale: float=1.0,
        ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * np.random.normal(size=score_t.shape)
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * np.sqrt(dt) * z

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = np.ones(x_t.shape[:-1])
        x_t_1 = x_t - perturb
        if center:
            com = np.sum(x_t_1, axis=-2) / np.sum(mask, axis=-1)[..., None]
            x_t_1 -= com[..., None, :]
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def conditional_var(self, t, use_torch=False):
        """Return conditional variance term of p(x_t | x_0).

        Args:
            t: Diffusion time value(s) in [0, 1].
            use_torch: Whether to use torch or numpy exponential ops.

        Returns:
            Scalar/array variance factor in the isotropic covariance.
        """
        if use_torch:
            return 1 - torch.exp(-self.marginal_b_t(t))
        return 1 - np.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t, use_torch=False, scale=False):
        """Evaluate conditional score of p(x_t | x_0) under the VP-SDE.

        Args:
            x_t: Noisy translations at time t.
            x_0: Clean translations used as conditioning signal.
            t: Diffusion time in [0, 1].
            use_torch: Whether to use torch ops for exponentials/variance.
            scale: Whether to scale x_t and x_0 before scoring.

        Returns:
            Score tensor/array with the same shape as x_t.
        """
        if use_torch:
            exp_fn = torch.exp
        else:
            exp_fn = np.exp
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        return -(x_t - exp_fn(-1/2*self.marginal_b_t(t)) * x_0) / self.conditional_var(t, use_torch=use_torch)
