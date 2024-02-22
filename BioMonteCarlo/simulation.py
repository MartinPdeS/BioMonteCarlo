#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

from typing import NoReturn

import numpy as np
from BioMonteCarlo.mesh import LayeredMesh
from BioMonteCarlo.photon import Photon
import matplotlib.pyplot as plt


class MonteCarloSimulator:
    def __init__(self, mesh: LayeredMesh, num_photons: int, initial_photon_direction=np.array([0.0, 0.0, 1.0])):
        """
        Initializes the Monte Carlo Simulator with a specified mesh structure, number of photons, and their initial direction.

        This simulator tracks the paths of multiple photons as they propagate through a layered mesh, simulating processes
        such as scattering, absorption, and reflection/transmission at layer boundaries. The simulator is capable of
        providing insights into photon-tissue interactions, which are crucial for applications like optical imaging and
        phototherapy.

        Parameters:
        -----------
        mesh : LayeredMesh
            The mesh representing the layered structure of the biological tissue. This structure defines the spatial
            distribution and optical properties of the tissue, including refractive indices, absorption coefficients,
            scattering coefficients, and anisotropy factors for each layer.

        num_photons : int
            The number of photons to simulate. This parameter determines the statistical accuracy of the simulation,
            with a higher number of photons typically leading to more accurate results.

        initial_photon_direction : np.ndarray, optional
            The initial direction of all photons at the start of the simulation, specified as a 3D unit vector.
            The default direction is along the positive z-axis (np.array([0.0, 0.0, 1.0])), which corresponds to
            photons entering the tissue perpendicular to its surface.

        Attributes:
        -----------
        photon_paths : list of lists
            A collection of photon paths, where each path is a list of 3D positions (np.ndarray) representing the
            trajectory of a photon through the tissue.

        absorption_profile : np.ndarray
            An array representing the total absorption in each layer of the mesh. This profile provides insight into
            where photons are being absorbed within the tissue, which is useful for applications like photodynamic therapy.
        """
        self.mesh = mesh
        self.num_photons = num_photons
        self.initial_photon_direction = initial_photon_direction
        self.photon_paths = []  # Store paths of all photons

        self.absorption_profile = np.zeros_like(mesh._layer_boundaries[:-1])  # Absorption per layer

    def run(self):
        """
        Executes the Monte Carlo simulation, propagating each photon through the tissue based on the defined mesh properties.

        During the simulation, each photon's path is influenced by scattering and absorption events determined by the
        optical properties of the tissue layers. The simulation tracks the trajectory and final state (absorbed, reflected,
        or transmitted) of each photon, updating the absorption profile of the tissue accordingly.
        """
        for _ in range(self.num_photons):
            photon = Photon(self.initial_photon_direction)
            while photon.alive:
                current_position = photon.path[-1]

                # Check if the photon is outside the mesh boundaries
                if current_position[2] < self.mesh._layer_boundaries[0] or current_position[2] > self.mesh._layer_boundaries[-1]:
                    photon.alive = False
                    continue

                layer_index = self.mesh.get_layer_index(current_position[2])
                layer_props = self.mesh._layer_properties[layer_index]

                step_size = -np.log(np.random.rand()) / layer_props['mu_s']
                photon.move(step_size)
                photon.check_weight()

                # Record absorption (simplified)
                self.absorption_profile[layer_index] += photon.weight * layer_props['mu_a'] * step_size
                photon.weight *= (1 - layer_props['mu_a'] * step_size)

            self.photon_paths.append(photon.path)

    def plot_photon_paths(self) -> NoReturn:
        """
        Visualizes the paths of a subset of simulated photons within the tissue, highlighting the depth of penetration and
        the interactions with layer boundaries.

        The method plots the z-component of the photon paths to illustrate their trajectories through the tissue. Layer
        boundaries are marked with horizontal lines, providing context for the photons' interactions with different tissue
        layers. This visualization aids in understanding photon scattering, absorption, and reflection/transmission behaviors.
        """
        plt.figure(figsize=(10, 6))

        # Plot photon paths
        for path in self.photon_paths[:100]:  # Plot paths of the first 100 photons
            z_coords = [pos[2] for pos in path]
            plt.plot(range(len(z_coords)), z_coords, alpha=0.5)  # Reduced alpha for better visibility of layer lines

        # Draw horizontal lines for mesh layer boundaries
        for boundary in self.mesh._layer_boundaries:
            plt.axhline(y=boundary, color='k', linestyle='--', linewidth=1)

        plt.xlabel('Step')
        plt.ylabel('Depth (z)')
        plt.title('Photon Paths in Tissue with Layer Delimitations')
        plt.show()


