# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import contextlib


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Temporarily seed NumPy PRNG and restore previous state.

    Args:
        seed: Base random seed; if None, no reseeding is applied.
        *addl_seeds: Optional extra values mixed into the effective seed.

    Returns:
        Context manager that yields control with deterministic NumPy state.
    """
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
