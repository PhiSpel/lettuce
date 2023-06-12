"""
Turbulent Mixing Layer flow
"""

import numpy as np

from lettuce.unit import UnitConversion
from lettuce.util import append_axes
from lettuce.boundary import BounceBackBoundary, EquilibriumBoundaryPU, AntiBounceBackOutlet


class TurbulentMixing2D(object):
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.shape = (resolution, resolution)
        self._mask = np.zeros(shape=self.shape, dtype=bool)
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    # def analytic_solution(self):
        # return []

    def initial_solution(self, grid):
        x, y = grid
        p = np.array([0 * x], dtype=float)  # no pressure anywhere
        ux = y >= np.max(y)/2  # "bottom" half (y-Axis is inverted) flows
        uy = np.zeros(self.shape)
        u = np.array([ux, uy], dtype=bool)  # .nonzero()  # move only bottom layer
        # u = np.array(p * u_char, dtype=bool)
        return p, u  # _char

    @property
    def mask(self):
        return self._mask

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        return []

    def _unit_vector(self, i=0):
        return np.eye(self.units.lattice.D)[i]


class TurbulentMixing3D(object):
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.shape = (resolution, resolution, resolution)
        self._mask = np.zeros(shape=self.shape, dtype=bool)
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    # def analytic_solution(self):
        # return []

    def initial_solution(self, grid):
        x, y, z = grid
        p = np.array([0 * x], dtype=float)  # no pressure anywhere
        ux = np.sin(2*np.pi*y)  # sinusoidal shaped flow
        uy = np.zeros(self.shape)
        uz = np.zeros(self.shape)
        u = np.array([ux, uy, uz], dtype=float)
        return p, u

    @property
    def mask(self):
        return self._mask

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution, endpoint=False)
        z = np.linspace(0, 1, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        return []

    def _unit_vector(self, i=0):
        return np.eye(self.units.lattice.D)[i]
