#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

from typing import NoReturn
import numpy as np
from dataclasses import dataclass
from BioMonteCarlo.henyey_greenstein import sample_henyey_greenstein


@dataclass
class Photon:
    """
    Represents a photon with position, direction, and other properties relevant to Monte Carlo simulations of light propagation.

    Attributes:
    -----------
    position : tuple of float
        The starting position of the photon, defaulting to the origin (0.0, 0.0, 0.0). The position is represented as a 3D vector.

    direction : tuple of float
        The initial direction of the photon's movement, defaulting to the positive z-direction (0.0, 0.0, 1.0). The direction is a unit vector.

    Methods:
    --------
    move(step_size):
        Moves the photon by a specified step size in its current direction, updating its position.

    scatter(g):
        Randomly changes the photon's direction based on the anisotropy factor g, simulating scattering events.

    absorb(mu_a, step_size):
        Updates the photon's weight based on the absorption coefficient mu_a and the step size, simulating energy loss due to absorption.

    check_weight(threshold):
        Checks if the photon's weight is below a specified threshold, marking it as terminated if true.
    """
    position: tuple = (0.0, 0.0, 1.0)
    """ Starting at the origin """
    direction: tuple = (0.0, 0.0, 1.0)
    """ Initially moving in the z-direction """

    def __post_init__(self):
        self.position = np.asarray(self.position)
        self.direction = np.asarray(self.direction)

        self.path = [self.position]  # Initialize path with the starting position
        self.weight = 1.0  # Initial weight
        self.alive = True

    def move(self, step_size: float) -> NoReturn:
        """
        Updates the photon's position based on its current direction and a given step size.

        Parameters:
        -----------
        step_size : float
            The distance the photon should move along its current direction vector.

        Equation:
        ---------
        new_position = current_position + direction * step_size
        """
        new_position = self.path[-1] + self.direction * step_size
        self.path.append(new_position)

    def scatter(self, g: float) -> NoReturn:
        """
        Updates the photon's direction based on the anisotropy factor g, simulating a scattering event.

        Parameters:
        -----------
        g : float
            The anisotropy factor, which determines the scattering angle distribution. g = 1 means forward scattering,
            g = -1 means backward scattering, and g = 0 means isotropic scattering.

        Equation:
        ---------
        theta = arccos(2 * rand() - 1)
        phi = 2 * pi * rand()
        direction = [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]
        """
        theta = sample_henyey_greenstein(g)

        phi = 2 * np.pi * np.random.rand()  # Random azimuthal angle

        self.direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

    def absorb(self, mu_a: float, step_size: float) -> NoReturn:
        """
        Updates the photon's weight to simulate absorption based on the absorption coefficient and step size.

        Parameters:
        -----------
        mu_a : float
            The absorption coefficient, representing the probability of photon absorption per unit distance.

        step_size : float
            The distance the photon moved during the current step.

        Equation:
        ---------
        absorbed = weight * (1 - exp(-mu_a * step_size))
        new_weight = weight - absorbed
        """
        absorbed = self.weight * (1 - np.exp(-mu_a * step_size))
        self.weight -= absorbed

    def check_weight(self, threshold: float = 0.001) -> NoReturn:
        """
        Checks if the photon's weight is below a specified threshold, marking the photon as terminated if true.

        Parameters:
        -----------
        threshold : float, optional
            The weight below which the photon is considered to be effectively absorbed and terminated.

        If weight < threshold, the photon's alive attribute is set to False, terminating its propagation.
        """
        if self.weight < threshold:
            self.alive = False


# -
