import torch
from lettuce.unit import UnitConversion
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, EquilibriumOutletP, AntiBounceBackOutlet
from lettuce.util import append_axes
import lettuce as lt

"""Linear Function y = a*x"""
x = torch.ones(5, requires_grad=True)
x = x * 2  # input tensor
a = 2
y = torch.sum(a*x)
x.retain_grad()
y.backward()
print(x.grad)
print(a)

"""More complex Function y = a*x"""
x = torch.ones(5, requires_grad=True)  # input tensor
a = torch.randn(5)
b = torch.randn(5, 3)
y = torch.sum(a*x*x) + torch.sum(torch.matmul(x, b))
y.backward()
print(x.grad)


class ObstacleAutograd:
    """Obstacle Flow"""
    def __init__(self, shape, reynolds_number, mach_number, lattice, domain_length_x, char_length=1, char_velocity=1):
        if len(shape) != lattice.D:
            raise ValueError(f"{lattice.D}-dimensional lattice requires {lattice.D}-dimensional `shape`")
        self.shape = shape
        char_length_lu = shape[0] / domain_length_x * char_length
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=char_length,
            characteristic_velocity_pu=char_velocity
        )
        self._mask = torch.zeros(self.shape, dtype=torch.bool)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, torch.Tensor) and m.shape == self.shape
        self._mask = m.astype(bool)

    def initial_solution(self, x):
        p = torch.zeros_like(x[0], dtype=torch.float)[None, ...]
        u_char = self.units.characteristic_velocity_pu * self._unit_vector()
        u_char = append_axes(u_char, self.units.lattice.D)
        u = self.mask * u_char * 0  # (1 - self.mask) * u_char * 0
        # p = np.zeros_like(x[0], dtype=float)[None, ...]
        # u = np.zeros_like(x[0], dtype=float)
        return p, u

    @property
    def grid(self):
        xyz = tuple(self.units.convert_length_to_pu(torch.arange(n)) for n in self.shape)
        return torch.meshgrid(*xyz, indexing='ij')

    @property
    def boundaries(self):
        x = self.grid[0]
        return [
            EquilibriumBoundaryPU(
                torch.abs(x) < 1e-6, self.units.lattice, self.units,
                self.units.characteristic_velocity_pu * self._unit_vector()
            ),
            EquilibriumOutletP(self.units.lattice, self._unit_vector().tolist()),
            BounceBackBoundary(self.mask, self.units.lattice)
        ]

    def _unit_vector(self, i=0):
        return torch.eye(self.units.lattice.D)[i]

    @boundaries.setter
    def boundaries(self, value):
        self._boundaries = value



lattice = lt.Lattice(lt.D2Q9, device="cpu")
Ma = torch.ones(1, requires_grad=True) * 0.1
flow = ObstacleAutograd(
    shape=(101, 51),
    reynolds_number=100,
    mach_number=Ma,
    lattice=lattice,
    domain_length_x=10.1
)
x, y = flow.grid

    # TODO: This approach does not work. Autograd needs float, not bool. --> Use partially saturated boundary
    # TODO: For now, try to change inflow or sth. else
    # r = torch.autograd.Variable(torch.ones(1), requires_grad=True)*r0
    # condition = torch.autograd.Variable(torch.where(torch.sqrt((x-r)**2+(y-r)**2) < 1., dtype=torch.float), requires_grad=True)
    # try:
    #     torch.sum(condition).backward()
    #     print(condition.grad)
    # except Exception as er:
    #     print(er)
    # flow.mask[condition] = 1
    # try:
    #     torch.sum(flow.mask).backward()
    #     print(r.grad)
    # except Exception as er:
    #     print(er)
collision = lt.KBCCollision2D(lattice, tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)

# energy = lt.IncompressibleKineticEnergy(lattice, flow)
# simulation.reporters.append(lt.ObservableReporter(energy, interval=1000, out=None))
simulation.initialize_f_neq()
mlups = simulation.step(100)
energy = flow.units.convert_incompressible_energy_to_pu(torch.sum(lattice.incompressible_energy(simulation.f)))
# energy.backward()
Ma.retain_grad()
torch.sum(simulation.f).backward()
print(Ma.grad)

