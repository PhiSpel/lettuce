import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
import torch

"""
For descriptions of the initialization, refer to 'example4beginners.py'.
"""

lattice = lt.Lattice(lt.D2Q9, device="cpu", dtype=torch.float32)
nx = 100
ny = 100
Re = 100
Ma = 0.1
ly = 1

flow = lt.Obstacle((nx, ny), reynolds_number=Re, mach_number=Ma, lattice=lattice, domain_length_x=ly)

x, y = flow.grid
r = .1      # radius
x_c = 0.5   # center along x
y_c = 0.5   # center along y
flow.mask = ((x + x_c) ** 2 + (y + y_c) ** 2) < (r ** 2)

collision = lt.KBCCollision2D(lattice, tau=flow.units.relaxation_parameter_lu)

streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)

simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=100, filename_base="./output"))

"""
We now add a reporter which we access later. The output can be written to files specified by out="reporter.txt"
"""
energy = lt.IncompressibleKineticEnergy(lattice, flow)
simulation.reporters.append(lt.ObservableReporter(energy, interval=1000, out=None))

"""
Now, we do not just run the whole simulation for 10,000 steps, but regard the energy convergence every 1000 steps.
"""
nmax = 10000; ntest = 1000
simulation.initialize_f_neq()
it = 0; i = 0; mlups = 0
energy_old = 1; energy_new = 1
while it <= nmax:
    i += 1
    it += ntest
    mlups += simulation.step(ntest)
    energy_new = energy(simulation.f).cpu().mean().item()
    print(f"avg MLUPS: {mlups / i:.3f}, avg energy: {energy_new:.8f}")
    if not energy_new == energy_new:
        print("CRASHED!")
        break
    if abs(energy_new - energy_old)/energy_old < 1e-6:
        print(f"CONVERGED! To {(energy_new - energy_old)/energy_old:.8f} after {it} iterations.")
        break
    energy_old = energy_new

# mlups = simulation.step(num_steps=10000)  # mlups can be read, but does not need to be
# print("Performance in MLUPS:", mlups)

