"""
mixing layer flow
"""

import numpy as np

from lettuce.unit import UnitConversion
from lettuce.util import append_axes
from lettuce.boundary import BounceBackBoundary, EquilibriumBoundaryPU, AntiBounceBackOutlet

def randu(nx, ny, nz):
    # returns values between -1 and 1
    return (np.random.rand(nx, ny, nz)-0.5)*2


class MixingLayer3D(object):
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

    def initial_solution(self, x):
        p = np.array([0 * x[0]], dtype=float)
        nx, ny, nz = self.shape
        shearlayerthickness = 1
        amplitude = 1
        period = 100
        sine = np.sin(period*x[0]) + np.sin(period*x[1]) + np.sin(period*x[2])
        centering = np.exp(-pow(x[1]/(2*shearlayerthickness), 2)) * amplitude
        ux = np.tanh(x[1]/(2*shearlayerthickness)) + randu(nx, ny, nz) * centering
        # double rd_sin = (rand() / (RAND_MAX) + sin(period*x(0)) + sin(period*x(1)) + sin(period*x(2))) * exp(-pow((x(1)+0.3)/(2*shearLayerThickness),2)) * amplitude;
        uy = randu(nx, ny, nz) * centering
        uz = randu(nx, ny, nz) * centering
        u = np.array([ux, uy, uz], dtype=float)
        return p, u

    @property
    def mask(self):
        return self._mask

    @property
    def grid(self):
        x = np.linspace(-1, 1, num=self.resolution, endpoint=False)
        y = np.linspace(-1, 1, num=self.resolution, endpoint=False)
        z = np.linspace(-1, 1, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        x, y, z = self.grid
        top = np.zeros(np.shape(y), dtype=bool)
        bottom = np.zeros(np.shape(y), dtype=bool)
        bottom[:, 0, :] = True  # bottom
        top[:, -1, :] = True  # top
        return [
            # bounce back walls
            # BounceBackBoundary(boundary, self.units.lattice),
            # moving fluid on top# moving bounce back top
            EquilibriumBoundaryPU(top, self.units.lattice, self.units, np.array([1.0, 0.0, 0.0])),
            EquilibriumBoundaryPU(bottom, self.units.lattice, self.units, np.array([-1.0, 0.0, 0.0])),
        ]
