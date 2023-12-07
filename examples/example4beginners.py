import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
import torch

"""
Lattice definitions.

The lattice is defined by the used stencil (mostly D2Q9, D3Q19, or D3Q27 - the more the expensivier).
The lattice is stored on CPU (device="cpu") or GPU (device="cuda").
"""
lattice = lt.Lattice(lt.D2Q9, device="cuda", dtype=torch.float32)

"""
Flow definitions.

We need 
1. the resolution in x and y direction
2. the Reynolds number (i.e., how fast the flow behaves compared to the object's length and fluid's viscosity)
3. the Mach number (i.e., how fast the flow is compared to speed of sound; Ma=0.3 is stable, above is discouraged)
4. the physical domain length in x-direction (this defines how lattice units scale to physical units)
to initialize the Obstacle flow object.
"""
nx = 100
ny = 100
Re = 100
Ma = 0.1
ly = 1

flow = lt.Obstacle((nx, ny), reynolds_number=Re, mach_number=Ma, lattice=lattice, domain_length_x=ly)

"""
Collision definition.

The collision is usually BGK (low dissipation, but may be unstable) or KBC (higher dissipation, but generally stable).
BGK is preferred for converging flows, KBC is preferred for driven flows in smaller domains
    (where energy conversation plays a smaller role, but gradients may be higher).
"""
collision = lt.KBCCollision2D(lattice, tau=flow.units.relaxation_parameter_lu)

"""
Streaming and simulation object setup.
"""
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)

"""
Reporters.

Reporter objects may be used to extract information later on or during the simulation.
They can be created as objects when required later (
"""
Energy = lt.IncompressibleKineticEnergy(lattice, flow)
reporter = lt.ObservableReporter(Energy, interval=1000, out=None)
simulation.reporters.append(reporter)
simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=100, filename_base="./output"))

simulation.initialize_f_neq()
mlups = simulation.step(num_steps=10000)  # mlups can be read, but does not need to be
print("Performance in MLUPS:", mlups)

