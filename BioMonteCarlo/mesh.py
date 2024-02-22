#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, NoReturn


class LayeredMesh:
    def __init__(self):
        """
        Initializes an empty LayeredMesh object with no layers.
        Layers are expected to be added using the add_layer method.
        """
        self._layer_boundaries = np.array([])  # Private attribute
        self._layer_properties = []  # Private attribute

    def add_layer(self, boundary, **properties):
        """
        Adds a new layer to the mesh with specified boundary and optical properties.

        Parameters:
        -----------
        boundary : float
            The z-coordinate that marks the upper boundary of the new layer. It must be greater than
            the last boundary in the `_layer_boundaries` array.

        properties : dict
            A dictionary containing the 'n', 'mu_s', 'mu_a', and 'g' properties of the new layer.

        Raises:
        -------
        ValueError
            If the new layer boundary is not greater than the last boundary or if the properties
            dictionary does not contain all required keys.
        """
        if self._layer_boundaries.size > 0 and boundary <= self._layer_boundaries[-1]:
            raise ValueError("New layer boundary must be greater than the last boundary.")
        if not all(key in properties for key in ('n', 'mu_s', 'mu_a', 'g')):
            raise ValueError("New layer must specify 'n', 'mu_s', 'mu_a', and 'g' properties.")

        if self._layer_boundaries.size == 0:
            self._layer_boundaries = np.array([boundary])
        else:
            self._layer_boundaries = np.append(self._layer_boundaries, boundary)
        self._layer_properties.append(properties)

    def get_layer_index(self, z):
        """
        Determines the index of the layer that contains the given z-coordinate.

        Parameters:
        -----------
        z : float
            The z-coordinate for which to determine the layer index.

        Returns:
        --------
        int
            The index of the layer containing the given z-coordinate. Returns -1 if outside any layer.
        """
        if z < self._layer_boundaries[0] or z > self._layer_boundaries[-1]:
            return -1  # z-coordinate is outside the mesh
        return np.searchsorted(self._layer_boundaries, z, side='right') - 1

    def get_properties_at(self, z):
        """
        Retrieves the optical properties at a given z-coordinate.

        Parameters:
        -----------
        z : float
            The z-coordinate at which to retrieve the optical properties.

        Returns:
        --------
        dict or None
            A dictionary containing the 'n', 'mu_s', 'mu_a', and 'g' values at the specified z-coordinate,
            or None if z-coordinate is outside any layer.
        """
        layer_index = self.get_layer_index(z)
        if layer_index == -1:
            return None
        return self._layer_properties[layer_index]

    def plot(self) -> NoReturn:
        """
        Generates a plot showing the optical properties across the depth of the mesh.
        Each optical property is plotted in a separate subplot.
        """
        fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
        properties = ['n', 'mu_s', 'mu_a', 'g']
        titles = ['Refractive Index (n)', 'Scattering Coefficient (μs)', 'Absorption Coefficient (μa)', 'Anisotropy (g)']

        for i, prop in enumerate(properties):
            values = [layer[prop] for layer in self.layer_properties]
            axs[i].step(values, self.layer_boundaries[:-1], where='post', label=prop)
            axs[i].set_title(titles[i])
            axs[i].set_xlabel(titles[i])
            axs[i].grid(True)

        axs[0].set_ylabel('Depth (z)')
        plt.suptitle('Layered Mesh Optical Properties')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# -
