#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import matplotlib.pyplot as plt
import numpy as np


class Analysis:
    def __init__(self, simulator):
        """
        Initializes the Analysis class with results from a MonteCarloSimulator instance to provide
        various analysis and visualization tools for understanding the simulation outcomes.

        Parameters:
        -----------
        simulator : MonteCarloSimulator
            An instance of the MonteCarloSimulator class after the simulation has been run. This instance
            should contain the simulation results, including photon paths and absorption profiles, which
            are necessary for the analysis.
        """
        self.simulator = simulator

    def absorption_profile(self):
        """
        Plots the absorption profile across the tissue depth, showing the total absorption in each layer.
        This visualization helps understand where photons are being absorbed within the tissue layers.

        The method generates a bar plot where each bar represents a layer, and the bar's height indicates
        the total absorption in that layer.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(self.simulator.absorption_profile)), self.simulator.absorption_profile, align='center')
        plt.xlabel('Layer')
        plt.ylabel('Absorption')
        plt.title('Absorption Profile Across Tissue Layers')
        plt.xticks(range(len(self.simulator.absorption_profile)), labels=[f'Layer {i+1}' for i in range(len(self.simulator.absorption_profile))])
        plt.show()

    def reflection_transmission(self):
        """
        Calculates and prints the overall reflection and transmission rates based on the photon paths.
        Reflection rate is determined by the number of photons that are reflected back from the tissue surface,
        and transmission rate is calculated from the photons that pass through the entire tissue.

        The method prints the reflection and transmission rates as percentages.
        """
        reflected = sum(1 for path in self.simulator.photon_paths if path[-1][2] < self.simulator.mesh._layer_boundaries[0])
        transmitted = sum(1 for path in self.simulator.photon_paths if path[-1][2] > self.simulator.mesh._layer_boundaries[-1])
        total_photons = len(self.simulator.photon_paths)

        reflection_rate = reflected / total_photons
        transmission_rate = transmitted / total_photons

        print(f"Reflection Rate: {reflection_rate:.2f}")
        print(f"Transmission Rate: {transmission_rate:.2f}")

    def penetration_depth_distribution(self):
        """
        Plots the distribution of the maximum penetration depth of all photons. This visualization helps
        understand how deeply photons can penetrate into the tissue and how the penetration depths are distributed.

        The method generates a histogram showing the frequency of photons reaching various depths within the tissue.
        """
        max_depths = [max(path, key=lambda p: p[2])[2] for path in self.simulator.photon_paths]
        plt.figure(figsize=(10, 6))
        plt.hist(max_depths, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Penetration Depth')
        plt.ylabel('Frequency')
        plt.title('Distribution of Photon Penetration Depths')
        plt.show()

    def plot_layered_absorption_map(self):
        """
        Plots a simplified 2D map of absorption where each row represents a layer in the mesh, and the color
        intensity indicates the level of absorption in that layer. This method assumes uniform absorption across
        the x-y plane for each layer.

        The method generates a heatmap where each row corresponds to a layer in the mesh, providing a visual
        representation of absorption in different tissue layers.
        """

        # Assuming uniform absorption across the x-y plane for each layer
        # The absorption_profile from the simulator gives the total absorption in each layer
        num_layers = len(self.simulator.absorption_profile)
        absorption_map = np.zeros((num_layers, 10))  # Create a 2D array with 10 columns for visualization

        # Fill each row (layer) in the absorption_map with the total absorption value from that layer
        for i, absorption in enumerate(self.simulator.absorption_profile):
            absorption_map[i, :] = absorption

        plt.figure(figsize=(10, 6))
        plt.imshow(absorption_map, cmap='hot', aspect='auto')
        plt.colorbar(label='Absorption')
        plt.xlabel('Arbitrary Units')
        plt.ylabel('Layer')
        plt.title('Layered Absorption Map in Tissue')
        plt.yticks(range(num_layers), labels=[f'Layer {i+1}' for i in range(num_layers)])
        plt.show()
# -
