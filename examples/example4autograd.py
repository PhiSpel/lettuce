import torch
from lettuce.unit import UnitConversion
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, EquilibriumOutletP, AntiBounceBackOutlet
from lettuce.flows.obstacleCylinder import ObstacleCylinder
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

lattice = lt.Lattice(lt.D2Q9, device="cpu", use_native=False)
Ma = torch.ones(1, requires_grad=True) * 0.1
flow = ObstacleCylinder(
    shape=(20, 10),
    reynolds_number=100,
    mach_number=Ma,
    lattice=lattice
)
x, y = flow.grid
collision = lt.KBCCollision2D(lattice, tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)

# energy = lt.IncompressibleKineticEnergy(lattice, flow)
# simulation.reporters.append(lt.ObservableReporter(energy, interval=1000, out=None))
simulation.initialize_f_neq()
mlups = simulation.step(100)
energy = flow.units.convert_incompressible_energy_to_pu(torch.sum(lattice.incompressible_energy(simulation.f)))
Ma.retain_grad()
# torch.sum(simulation.f).backward()
energy.backward()
print(Ma.grad)
