import lettuce as lt
import numpy as np


class FlatPlate(object):
    def __init__(self, shape, reynolds_number, mach_number, lattice, domain_length_x=1, char_length=1, char_velocity=1):
        self.shape = shape
        self.resolution = shape[0]
        self.lattice = lattice
        char_length_lu = shape[0] / domain_length_x * char_length
        self.units = lt.UnitConversion(lattice, reynolds_number=reynolds_number, mach_number=mach_number,
                                       characteristic_length_lu=char_length_lu, characteristic_length_pu=char_length,
                                       characteristic_velocity_pu=char_velocity)

    def initial_solution(self, x):
        nx, ny, nz = self.shape
        u1 = (np.random.rand(nx, ny, nz) - 0.5) * 2
        u2 = (np.random.rand(nx, ny, nz) - 0.5) * 2
        u3 = (np.random.rand(nx, ny, nz) - 0.5) * 2
        u = np.array([u1, u2, u3])
        return np.array([0 * x[0]], dtype=float), u

    @property
    def grid(self):
        xyz = tuple(self.units.convert_length_to_pu(np.arange(n)) for n in self.shape)
        return np.meshgrid(*xyz, indexing='ij')

    @property
    def boundaries(self):
        plate = np.zeros(self.shape, dtype=bool)
        top = np.zeros(self.shape, dtype=bool)
        inflow = np.zeros(self.shape, dtype=bool)
        outflow = np.zeros(self.shape, dtype=bool)
        plate[:, 0, :] = True
        top[:, -1, :] = True
        inflow[0, :, :] = True
        outflow[-1, :, :] = True
        return [lt.BounceBackBoundary(plate, self.units.lattice),
                lt.SlipBoundary(top, self.units.lattice, 1),
                lt.EquilibriumBoundaryPU(inflow, self.units.lattice, self.units,
                                         self.units.characteristic_velocity_pu * self._unit_vector()),
                lt.EquilibriumOutletP(self.units.lattice, self._unit_vector().tolist())
                ]

    def _unit_vector(self, i=0):
        return np.eye(self.units.lattice.D)[i]
