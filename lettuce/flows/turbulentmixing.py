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
        x, y = self.grid
        ktop = np.zeros(np.shape(y), dtype=bool)
        kbottom = ktop
        kleft = ktop
        ktop[:, 0] = True  # top is at y = 0
        kbottom[:, -1] = True
        kleft[0, :] = True
        kleftbottomhalf = kleft & (y >= np.max(y)/2)
        klefttophalf = kleft & (y < np.max(y)/2)
        kright = ktop
        kright[-1, :] = True
        krightbottomhalf = kright & (y >= np.max(y)/2)
        krighttophalf = kright & (y < np.max(y)/2)
        return [
            # bounce back top
            BounceBackBoundary(ktop, self.units.lattice),
            # moving bounce back bottom
            EquilibriumBoundaryPU(kbottom, self.units.lattice, self.units, np.array([1.0, 0.0])),
            # Input flow on left bottom
            EquilibriumBoundaryPU(kleftbottomhalf, self.units.lattice, self.units, np.array([1.0, 0.0])),
            # Bounce back on left top
            EquilibriumBoundaryPU(klefttophalf, self.units.lattice, self.units, np.array([0.0, 0.0])),
            # repeat on right
            EquilibriumBoundaryPU(krightbottomhalf, self.units.lattice, self.units, np.array([1.0, 0.0])),
            EquilibriumBoundaryPU(krighttophalf, self.units.lattice, self.units, np.array([0.0, 0.0])),
            # equilibrium otherwise
            # AntiBounceBackOutlet(self.units.lattice, [-1, 0])
        ]

    def _unit_vector(self, i=0):
        return np.eye(self.units.lattice.D)[i]
