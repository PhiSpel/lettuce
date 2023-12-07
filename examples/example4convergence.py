import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
import torch

"""
For descriptions of the initialization, refer to 'example4beginners.py'.
"""

lattice = lt.Lattice(lt.D2Q9, device="cpu", dtype=torch.float32)
flow = lt.TaylorGreenVortex2D(resolution=256, reynolds_number=100, mach_number=0.05, lattice=lattice)
collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)

Energy = lt.IncompressibleKineticEnergy(lattice, flow)
reporter = lt.ObservableReporter(Energy, interval=1000, out=None)
simulation.reporters.append(reporter)
simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=100, filename_base="./output"))

"""
Here, we do not just run the simulation, but regard the convergence every so often.
"""

simulation.initialize_f_neq()
mlups = simulation.step(num_steps=10000)  # mlups can be read, but does not need to be
print("Performance in MLUPS:", mlups)

