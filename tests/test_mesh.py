#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import pytest
from BioMonteCarlo.mesh import LayeredMesh


def test_initialization():
    layer_boundaries = [0, 1.5, 3.0]
    layer_properties = [
        {'n': 1.0, 'mu_s': 10, 'mu_a': 0.1, 'g': 0.9},
        {'n': 1.4, 'mu_s': 5, 'mu_a': 0.05, 'g': 0.8}
    ]

    mesh = LayeredMesh(layer_boundaries, layer_properties)

    assert len(mesh.layer_boundaries) == len(layer_boundaries)
    assert len(mesh.layer_properties) == len(layer_properties)


def test_add_layer():
    mesh = LayeredMesh([0, 1.5], [{'n': 1.0, 'mu_s': 10, 'mu_a': 0.1, 'g': 0.9}])
    mesh.add_layer(3.0, {'n': 1.4, 'mu_s': 5, 'mu_a': 0.05, 'g': 0.8})

    assert len(mesh.layer_boundaries) == 3
    assert mesh.layer_boundaries[-1] == 3.0
    assert len(mesh.layer_properties) == 2
    assert mesh.layer_properties[-1]['n'] == 1.4


def test_get_properties_at():
    mesh = LayeredMesh([0, 1.5, 3.0], [
        {'n': 1.0, 'mu_s': 10, 'mu_a': 0.1, 'g': 0.9},
        {'n': 1.4, 'mu_s': 5, 'mu_a': 0.05, 'g': 0.8}
    ])

    properties = mesh.get_properties_at(2.0)  # Should be in the second layer
    assert properties['n'] == 1.4
    assert properties['mu_s'] == 5
    assert properties['mu_a'] == 0.05
    assert properties['g'] == 0.8


def test_invalid_layer_properties():
    with pytest.raises(ValueError):
        LayeredMesh([0, 1.5], [{'n': 1.0, 'mu_a': 0.1, 'g': 0.9}])  # Missing 'mu_s'


# -
