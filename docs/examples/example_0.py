#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

from BioMonteCarlo.mesh import LayeredMesh
from BioMonteCarlo.simulation import MonteCarloSimulator

# Example usage
layer_boundaries = [0, 1.5, 3.0]

mesh = LayeredMesh()

mesh.add_layer(0, n=1.0, mu_s=10, mu_a=0.1, g=0.9)

mesh.add_layer(1.5, n=1.33, mu_s=5, mu_a=0.1, g=0.9)

mesh.add_layer(3.0, n=1.57, mu_s=5, mu_a=0.1, g=0.9)

mesh.add_layer(5, n=1.4, mu_s=10, mu_a=10.1, g=0.9)

mesh.add_layer(15, n=1.4, mu_s=10, mu_a=100.1, g=0.9)

simulator = MonteCarloSimulator(mesh, 100, initial_photon_direction=[0, 0, 1])
simulator.run()

# Plot the paths of the first 100 photons
simulator.plot_photon_paths()

# Plot the absorption profile across layers
simulator.plot_absorption_profile()
# -