import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np

nreport = 500
Ma = 0.1
res = 64
Re = 1000

lattice = lt.Lattice(lt.D2Q9, device="cuda", use_native=False)
lattice2 = lt.Lattice(lt.D2Q9, device="cuda", use_native=False)
flow = lt.PoiseuilleFlow2D(res, Re, Ma, lattice, initialize_with_zeros=False)
flow2 = lt.PoiseuilleFlow2DHalf(res, Re, Ma, lattice2, initialize_with_zeros=False)
collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
collision2 = lt.BGKCollision(lattice2, tau=flow2.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
streaming2 = lt.StandardStreaming(lattice2)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
simulation2 = lt.Simulation(flow=flow2, lattice=lattice2, collision=collision2, streaming=streaming2)

Energy = lt.IncompressibleKineticEnergy(lattice, flow)
reporter = lt.ObservableReporter(Energy, interval=1000, out=None)
simulation.reporters.append(reporter)
simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=nreport, filename_base="./data/poiseuille"))

Energy2 = lt.IncompressibleKineticEnergy(lattice, flow2)
reporter2 = lt.ObservableReporter(Energy2, interval=1000, out=None)
simulation2.reporters.append(reporter2)
simulation2.reporters.append(lt.VTKReporter(lattice, flow2, interval=nreport, filename_base="./data/poiseuille_half"))

simulation.initialize_f_neq()
mlups = simulation.step(num_steps=10000)
print("Performance in MLUPS:", mlups)

simulation2.initialize_f_neq()
mlups = simulation2.step(num_steps=10000)
print("Performance in MLUPS:", mlups)

energy = np.array(simulation.reporters[0].out)
energy2 = np.array(simulation2.reporters[0].out)
plt.plot(energy[:, 1], energy[:, 2], energy2[:, 1], energy2[:, 2])
plt.title('Kinetic energy')
plt.xlabel('Time')
plt.ylabel('Energy in physical units')
plt.show()

u = flow.units.convert_velocity_to_pu(lattice.u(simulation.f)).cpu().numpy()
u_norm = np.linalg.norm(u, axis=0)
plt.imshow(u_norm)
plt.show()

u = flow.units.convert_velocity_to_pu(lattice.u(simulation2.f)).cpu().numpy()
u_norm = np.linalg.norm(u, axis=0)
plt.imshow(u_norm)
plt.show()
