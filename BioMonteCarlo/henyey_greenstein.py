#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import numpy as np


def sample_henyey_greenstein(g: float) -> float:
    """
    Samples a scattering angle from the Henyey-Greenstein phase function using the inverse transform sampling method.

    Parameters:
    -----------
    g : float
        The anisotropy factor of the phase function.

    Returns:
    --------
    theta : float
        The sampled scattering angle.
    """
    if np.abs(g) < 1e-10:
        # Isotropic scattering for g close to 0
        costheta = 2 * np.random.uniform() - 1
    else:
        rnd = np.random.uniform()
        costheta = (1 + g**2 - ((1 - g**2) / (1 - g + 2 * g * rnd))**2) / (2 * g)

    return np.arccos(costheta)

# -
